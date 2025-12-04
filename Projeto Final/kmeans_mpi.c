#define _POSIX_C_SOURCE 199309L
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <math.h>

typedef struct {
    int* coords;
    int cluster_id;
} Point;

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

void assign_points_to_clusters_local(Point* local_points, Point* centroids, int local_M, int K, int D) {
    for (int i = 0; i < local_M; i++) {
        long long min_dist = LLONG_MAX;
        int best_cluster = -1;

        for (int j = 0; j < K; j++) {
            long long dist = euclidean_dist_sq(&local_points[i], &centroids[j], D);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }
        local_points[i].cluster_id = best_cluster;
    }
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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 6) {
        if (rank == 0) {
            fprintf(stderr, "Uso: %s <arquivo_dados> <M_pontos> <D_dimensoes> <K_clusters> <I_iteracoes>\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const char* filename = argv[1];
    const int M = atoi(argv[2]);
    const int D = atoi(argv[3]);
    const int K = atoi(argv[4]);
    const int I = atoi(argv[5]);

    if (M <= 0 || D <= 0 || K <= 0 || I <= 0 || K > M) {
        if (rank == 0) {
            fprintf(stderr, "Erro nos parâmetros. Verifique se M,D,K,I > 0 e K <= M.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int total_coords_size = (M + K) * D;
    int* all_coords = (int*)malloc(total_coords_size * sizeof(int));
    Point* points = (Point*)malloc(M * sizeof(Point));
    Point* centroids = (Point*)malloc(K * sizeof(Point));
    
    if (!all_coords || !points || !centroids) {
        fprintf(stderr, "Erro de alocação de memória.\n");
        free(all_coords); free(points); free(centroids);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    for (int i = 0; i < M; i++) {
        points[i].coords = &all_coords[i * D];
        points[i].cluster_id = -1;
    }
    for (int i = 0; i < K; i++) {
        centroids[i].coords = &all_coords[(M + i) * D];
    }
    int* centroids_coords_ptr = &all_coords[M * D];


    if(rank == 0){
        read_data_from_file(filename, points, M, D);
    }
    MPI_Bcast(all_coords, M * D, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0){
        initialize_centroids(points, centroids, M, K, D);
    }
    MPI_Bcast(centroids_coords_ptr, K * D, MPI_INT, 0, MPI_COMM_WORLD);
    

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int points_per_process = M / size;
    int local_start = rank * points_per_process;
    int local_end = (rank == size - 1) ? M : local_start + points_per_process;
    int local_M = local_end - local_start;

    const int REDUCTION_SIZE = K * (D + 1);
    long long* local_reduction_buffer = (long long*)calloc(REDUCTION_SIZE, sizeof(long long));
    long long* global_reduction_buffer = (long long*)malloc(REDUCTION_SIZE * sizeof(long long));
    
    if (!local_reduction_buffer || !global_reduction_buffer) {
        fprintf(stderr, "Erro de alocação de buffer de redução.\n");
        free(all_coords); free(points); free(centroids);
        free(local_reduction_buffer); free(global_reduction_buffer);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    for (int iter = 0; iter < I; iter++) {

        assign_points_to_clusters_local(points + local_start,
                                        centroids,
                                        local_M,
                                        K,
                                        D);

        memset(local_reduction_buffer, 0, REDUCTION_SIZE * sizeof(long long));

        long long* local_sums   = local_reduction_buffer;
        long long* local_counts = local_reduction_buffer + K * D;

        for (int i = 0; i < local_M; i++) {
            int c = (points + local_start)[i].cluster_id;
            local_counts[c]++;
            for (int j = 0; j < D; j++)
                local_sums[c * D + j] += (points + local_start)[i].coords[j];
        }

        MPI_Allreduce(local_reduction_buffer,
                      global_reduction_buffer,
                      REDUCTION_SIZE,
                      MPI_LONG_LONG,
                      MPI_SUM,
                      MPI_COMM_WORLD);

        long long* global_sums   = global_reduction_buffer;
        long long* global_counts = global_reduction_buffer + K * D;

        for (int c = 0; c < K; c++) {
            if (global_counts[c] > 0) {
                for (int j = 0; j < D; j++) {
                    centroids[c].coords[j] =
                        (int)(global_sums[c * D + j] / global_counts[c]);
                }
            }
        }

    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_taken = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);

    if(rank == 0){
        print_time_and_checksum(centroids, K, D, time_taken);
    }

    free(all_coords);
    free(points);
    free(centroids);
    free(local_reduction_buffer);
    free(global_reduction_buffer);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
