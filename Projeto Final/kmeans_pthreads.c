#define _POSIX_C_SOURCE 199309L  // Necessário para CLOCK_MONOTONIC
#include <limits.h>              // Para LLONG_MAX
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <time.h>  // Header correto para clock_gettime e struct timespec

int thread_count;
pthread_mutex_t mutex;

char* filename;  // Nome do arquivo de dados
int M;   // Número de pontos
int D;   // Número de dimensões
int K;   // Número de clusters
int I;   // Número de i;

long long **global_cluster_sums;
int **global_cluster_counts;

// Estrutura para representar um ponto no espaço D-dimensional
typedef struct {
  int* coords;     // Vetor de coordenadas inteiras
  int cluster_id;  // ID do cluster ao qual o ponto pertence
}Point;

static pthread_barrier_t barrier;

  // --- Alocação de Memória ---
  int* all_coords;
  Point* points = NULL;
  Point* centroids = NULL;
  // ... (verificação de alocação) ...


void *thread(void* id){
  long tid = (long)id;

  for (int iter = 0; iter < I; iter++) {
    assign_points_to_clusters(points, centroids, M, K, D, tid);
    pthread_barrier_wait(&barrier);
    update_centroids(points, centroids, M, K, D, tid);
  }

  return NULL;
}

// --- Funções Utilitárias ---

/**
 * @brief Calcula a distância Euclidiana ao quadrado entre dois pontos com coordenadas inteiras.
 * Usa 'long long' para evitar overflow no cálculo da distância e da diferença.
 * @return A distância Euclidiana ao quadrado como um long long.
 */
long long euclidean_dist_sq(Point* p1, Point* p2, int D) {
  long long dist = 0;
  for (int i = 0; i < D; i++) {
    long long diff = (long long)p1->coords[i] - p2->coords[i];
    dist += diff * diff;
  }
  return dist;
}

// --- Funções Principais do K-Means ---

/**
 * @brief Lê os dados de pontos (inteiros) de um arquivo de texto.
 */
void read_data_from_file(const char* filename, Point* points, int M, int D) {
  FILE* file = fopen(filename, "r");
  if (file == NULL) {
    fprintf(stderr, "Erro: Não foi possível abrir o arquivo '%s'\n", filename);
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < D; j++) {
      if (fscanf(file, "%d", &points[i].coords[j]) != 1) {
        fprintf(stderr, "Erro: Arquivo de dados mal formatado ou incompleto.\n");
        fclose(file);
        exit(EXIT_FAILURE);
      }
    }
  }

  fclose(file);
}

/**
 * @brief Inicializa os centroides escolhendo K pontos aleatórios do dataset.
 */
void initialize_centroids(Point* points, Point* centroids, int M, int K, int D) {
  srand(42);  // Semente fixa para reprodutibilidade

  int* indices = (int*)malloc(M * sizeof(int));
  for (int i = 0; i < M; i++) {
    indices[i] = i;
  }

  for (int i = 0; i < M; i++) {
    int j = rand() % M;
    int temp = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
  }

  for (int i = 0; i < K; i++) {
    memcpy(centroids[i].coords, points[indices[i]].coords, D * sizeof(int));
  }

  free(indices);
}

/**
 * @brief Fase de Atribuição: Associa cada ponto ao cluster do centroide mais próximo.
 */
void assign_points_to_clusters(Point* points, Point* centroids, int M, int K, int D, long thread_id) {

  long block_size = M / thread_count;

  long i_inicial = thread_id*block_size;
  long i_final = i_inicial+block_size;

  if (thread_id == thread_count-1)    // se eu sou a última thread, fico com o 
    i_final = M;     

  for (int i = i_inicial; i < i_final; i++) {
    long long min_dist = LLONG_MAX;
    int best_cluster = -1;

    for (int j = 0; j < K; j++) {
      long long dist = euclidean_dist_sq(&points[i], &centroids[j], D);
      if (dist < min_dist) {
        min_dist = dist;
        best_cluster = j;
      }
    }
    points[i].cluster_id = best_cluster;
  }
}

/**
 * @brief Fase de Atualização: Recalcula a posição de cada centroide como a média
 * (usando divisão inteira) de todos os pontos atribuídos ao seu cluster.
 */
void update_centroids(Point* points, Point* centroids, int M, int K, int D, long thread_id) {
  long long* cluster_sums = global_cluster_sums[thread_id];
  int* cluster_counts = global_cluster_counts[thread_id];

  long i_inicial = (M * thread_id) / thread_count;
  long i_final   = (M * (thread_id + 1)) / thread_count;

  for (int i = i_inicial; i < i_final; i++) {
      int cluster_id = points[i].cluster_id;
      cluster_counts[cluster_id]++;
      for (int j = 0; j < D; j++) {
          cluster_sums[cluster_id * D + j] += points[i].coords[j];
      }
  }

  pthread_barrier_wait(&barrier);

  if (thread_id == 0) {
      for (int k = 0; k < K; k++) {
          long long total_sum[D];
          memset(total_sum, 0, sizeof(total_sum));
          int total_count = 0;

          for (int t = 0; t < thread_count; t++) {
              total_count += global_cluster_counts[t][k];
              for (int j = 0; j < D; j++) {
                  total_sum[j] += global_cluster_sums[t][k * D + j];
              }
          }

          if (total_count > 0) {
              for (int j = 0; j < D; j++) {
                  centroids[k].coords[j] = total_sum[j] / total_count;
              }
          }
      }

      for (int t = 0; t < thread_count; t++) {
          memset(global_cluster_sums[t], 0, K * D * sizeof(long long));
          memset(global_cluster_counts[t], 0, K * sizeof(int));
      }
  }

  pthread_barrier_wait(&barrier);
}


/**
 * @brief Imprime os resultados finais e o checksum (como long long).
 */
void print_results(Point* centroids, int K, int D) {
  printf("--- Centroides Finais ---\n");
  long long checksum = 0;
  for (int i = 0; i < K; i++) {
    printf("Centroide %d: [", i);
    for (int j = 0; j < D; j++) {
      printf("%d", centroids[i].coords[j]);
      if (j < D - 1) printf(", ");
      checksum += centroids[i].coords[j];
    }
    printf("]\n");
  }
  printf("\n--- Checksum ---\n");
  printf("%lld\n", checksum);  // %lld para long long int
}

/**
 * @brief Calcula e imprime o tempo de execução e o checksum final.
 * A saída é formatada para ser facilmente lida por scripts:
 * Linha 1: Tempo de execução em segundos (double)
 * Linha 2: Checksum final (long long)
 */
void print_time_and_checksum(Point* centroids, int K, int D, double exec_time) {
  long long checksum = 0;
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < D; j++) {
      checksum += centroids[i].coords[j];
    }
  }
  // Saída formatada para o avaliador
  printf("%lf\n", exec_time);
  printf("%lld\n", checksum);
}

// --- Função Principal ---

int main(int argc, char* argv[]) {
  pthread_t *thread_handles;
  pthread_mutex_init(&mutex, NULL);

  thread_count = 8;

  filename = argv[1];  // Nome do arquivo de dados
  M = atoi(argv[2]);     // Número de pontos
  D = atoi(argv[3]);     // Número de dimensões
  K = atoi(argv[4]);     // Número de clusters
  I = atoi(argv[5]);     // Número de iterações

  if (M <= 0 || D <= 0 || K <= 0 || I <= 0 || K > M) {
    fprintf(stderr, "Erro nos parâmetros. Verifique se M,D,K,I > 0 e K <= M.\n");
    return EXIT_FAILURE;
  }

  global_cluster_sums = malloc(thread_count * sizeof(long long*));
  global_cluster_counts = malloc(thread_count * sizeof(int*));

  for (int t = 0; t < thread_count; t++) {
      global_cluster_sums[t] = calloc(K * D, sizeof(long long));
      global_cluster_counts[t] = calloc(K, sizeof(int));
  }
  
  thread_handles = malloc(thread_count*sizeof(pthread_t));
  
  pthread_barrier_init(&barrier, NULL, thread_count);
  
  // Validação e leitura dos argumentos de linha de comando
  if (argc != 6) {
    fprintf(stderr, "Uso: %s <arquivo_dados> <M_pontos> <D_dimensoes> <K_clusters> <I_iteracoes>\n", argv[0]);
    return EXIT_FAILURE;
  }

  

  // --- Alocação de Memória ---
  all_coords = (int*)malloc((M + K) * D * sizeof(int));
  points = (Point*)malloc(M * sizeof(Point));
  centroids = (Point*)malloc(K * sizeof(Point));
  // ... (verificação de alocação) ...
  for (int i = 0; i < M; i++) {
    points[i].coords = &all_coords[i * D];
  }
  for (int i = 0; i < K; i++) {
    centroids[i].coords = &all_coords[(M + i) * D];
  }

  // --- Preparação (Fora da medição de tempo) ---
  read_data_from_file(filename, points, M, D);
  initialize_centroids(points, centroids, M, K, D);
  int i;
  // --- Medição de Tempo do Algoritmo Principal ---
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);  // Inicia o cronômetro

  for (i=0; i<thread_count; i++){
    if (pthread_create(&thread_handles[i], NULL, thread, (void *)i) != 0){
      fprintf(stderr, "Nao consegui criar a thread\n"); exit(-1);
    }
  }
  
  for (i=0; i<thread_count; i++){
    pthread_join(thread_handles[i], NULL);
  }
  free(thread_handles);

  clock_gettime(CLOCK_MONOTONIC, &end);  // Para o cronômetro

  pthread_barrier_destroy(&barrier);

  // Calcula o tempo decorrido em segundos
  double time_taken = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);

  // --- Apresentação dos Resultados ---
  print_time_and_checksum(centroids, K, D, time_taken);
  pthread_mutex_destroy(&mutex);
  // --- Limpeza ---
  free(all_coords);
  free(points);
  free(centroids);

  return EXIT_SUCCESS;
}
