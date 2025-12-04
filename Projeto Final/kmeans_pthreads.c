#define _POSIX_C_SOURCE 200112L
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <time.h>

typedef struct {
    int* coords;
    int cluster_id;
}Point;

long long euclidean_dist_sq(Point* p1, Point* p2, int D);
void read_data_from_file(const char* filename, Point* points, int M, int D);
void initialize_centroids(Point* points, Point* centroids, int M, int K, int D);
void assign_points_to_clusters(Point* points, Point* centroids, int M, int K, int D, long thread_id);
void update_centroids(Point* points, Point* centroids, int M, int K, int D, long thread_id);
void print_results(Point* centroids, int K, int D);
void print_time_and_checksum(Point* centroids, int K, int D, double exec_time);

int thread_count;

char* filename;
int M;
int D;
int K;
int I;

long long **global_cluster_sums;
int **global_cluster_counts;

static pthread_barrier_t barrier;

int* all_coords;
Point* points = NULL;
Point* centroids = NULL;


void *thread(void* id){
    long tid = (long)id;

    for (int iter = 0; iter < I; iter++) {
        assign_points_to_clusters(points, centroids, M, K, D, tid);
        update_centroids(points, centroids, M, K, D, tid);
    }

    return NULL;
}

long long euclidean_dist_sq(Point* p1, Point* p2, int D) {
    long long dist = 0;
    for (int i = 0; i < D; i++) {
        long long diff = (long long)p1->coords[i] - p2->coords[i];
        dist += diff * diff;
    }
    return dist;
}

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

void initialize_centroids(Point* points, Point* centroids, int M, int K, int D) {
    srand(42);

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

void assign_points_to_clusters(Point* points, Point* centroids, int M, int K, int D, long thread_id) {

    long i_inicial = (M * thread_id) / thread_count;
    long i_final = (M * (thread_id + 1)) / thread_count;
    
    for (long i = i_inicial; i < i_final; i++) {
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

void update_centroids(Point* points, Point* centroids, int M, int K, int D, long thread_id) {
    long long* cluster_sums = global_cluster_sums[thread_id];
    int* cluster_counts = global_cluster_counts[thread_id];

    
    // Zera buffers locais antes de acumular a nova iteração
    memset(cluster_sums, 0, K * D * sizeof(long long));
    memset(cluster_counts, 0, K * sizeof(int));

    long i_inicial = (M * thread_id) / thread_count;
    long i_final = (M * (thread_id + 1)) / thread_count;

    for (long i = i_inicial; i < i_final; i++) {
        int cluster_id = points[i].cluster_id;
        cluster_counts[cluster_id]++;
        for (int j = 0; j < D; j++) {
            cluster_sums[cluster_id * D + j] += points[i].coords[j];
        }
    }

    // Barreira 1: Espera todas as threads terminarem a acumulação local
    pthread_barrier_wait(&barrier);

    if (thread_id == 0) {
        long long total_sum[D];
        int total_count;

        for (int k = 0; k < K; k++) {
            total_count = 0;
            memset(total_sum, 0, sizeof(total_sum));
            
            // Agregação serial de todos os buffers locais
            for (int t = 0; t < thread_count; t++) {
                total_count += global_cluster_counts[t][k];
                for (int j = 0; j < D; j++) {
                    total_sum[j] += global_cluster_sums[t][k * D + j];
                }
            }

            // Cálculo e atualização do centroide
            if (total_count > 0) {
                for (int j = 0; j < D; j++) {
                    centroids[k].coords[j] = (int)(total_sum[j] / total_count);
                }
            }
        }
    }

    // Barreira 2: Espera a thread 0 terminar de atualizar os centroides
    pthread_barrier_wait(&barrier);
}


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
    printf("%lld\n", checksum);
}

void print_time_and_checksum(Point* centroids, int K, int D, double exec_time) {
    long long checksum = 0;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < D; j++) {
            checksum += centroids[i].coords[j];
        }
    }
    printf("%lf\n", exec_time);
    printf("%lld\n", checksum);
}

int main(int argc, char* argv[]) {
    pthread_t *thread_handles;
    
    if (argc != 6) {
        fprintf(stderr, "Uso: %s <arquivo_dados> <M_pontos> <D_dimensoes> <K_clusters> <I_iteracoes>\n", argv[0]);
        return EXIT_FAILURE;
    }

    filename = argv[1];
    M = atoi(argv[2]);
    D = atoi(argv[3]);
    K = atoi(argv[4]);
    I = atoi(argv[5]);

    if (M <= 0 || D <= 0 || K <= 0 || I <= 0 || K > M) {
        fprintf(stderr, "Erro nos parâmetros. Verifique se M,D,K,I > 0 e K <= M.\n");
        return EXIT_FAILURE;
    }

    // Hardcoded thread count
    thread_count = 8;
    
    global_cluster_sums = malloc(thread_count * sizeof(long long*));
    global_cluster_counts = malloc(thread_count * sizeof(int*));

    for (int t = 0; t < thread_count; t++) {
        global_cluster_sums[t] = calloc(K * D, sizeof(long long));
        global_cluster_counts[t] = calloc(K, sizeof(int));
    }
    
    thread_handles = malloc(thread_count*sizeof(pthread_t));
    
    pthread_barrier_init(&barrier, NULL, thread_count);
    
    all_coords = (int*)malloc((M + K) * D * sizeof(int));
    points = (Point*)malloc(M * sizeof(Point));
    centroids = (Point*)malloc(K * sizeof(Point));
    
    for (int i = 0; i < M; i++) {
        points[i].coords = &all_coords[i * D];
    }
    for (int i = 0; i < K; i++) {
        centroids[i].coords = &all_coords[(M + i) * D];
    }

    read_data_from_file(filename, points, M, D);
    initialize_centroids(points, centroids, M, K, D);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (long i=0; i<thread_count; i++){
        if (pthread_create(&thread_handles[i], NULL, thread, (void *)i) != 0){
            fprintf(stderr, "Nao consegui criar a thread\n"); exit(-1);
        }
    }
    
    for (long i=0; i<thread_count; i++){
        pthread_join(thread_handles[i], NULL);
    }
    free(thread_handles);

    clock_gettime(CLOCK_MONOTONIC, &end);

    pthread_barrier_destroy(&barrier);

    double time_taken = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);

    print_time_and_checksum(centroids, K, D, time_taken);
    
    for (int t = 0; t < thread_count; t++) {
        free(global_cluster_sums[t]);
        free(global_cluster_counts[t]);
    }
    free(global_cluster_sums);
    free(global_cluster_counts);
    
    free(all_coords);
    free(points);
    free(centroids);


    return EXIT_SUCCESS;
}
