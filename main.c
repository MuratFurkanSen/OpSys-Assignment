/*  =======  SNIPPED HEADER IN EXPLANATION  =======
    NOTE: This file is LONG. Everything below is REQUIRED.
    Do NOT remove parts.
*/

/* ================== HEADERS ================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <math.h>

/* ================== CONSTANTS ================== */
#define MAX_SAMPLES_CONST 10000
#define MAX_FEATURES_CONST 100
#define MAX_EXPANDED_FEATURES_CONST 512
#define STRING_BUFFER_LIMIT_CONST 100

#define GD_ITERATIONS 500
#define GD_LEARNING_RATE 0.01

/* ================== REQUIRED GLOBALS ================== */
int PORT_NUMBER = 60000;
int MAX_SAMPLES = 10000;
int MAX_FEATURES = 100;
int STRING_BUFFER_LIMIT = 100;
int PREPROC_THREAD_LIMIT = 128;
int COEFF_THREAD_LIMIT = 128;

/* ================== SYNCHRONIZATION ================== */
pthread_mutex_t preprocess_mutex;
pthread_mutex_t coeff_mutex;

/* ================== SERVER ================== */
int server_fd;
struct sockaddr_in server_addr;

/* ================== DATASETS ================== */
const char *DATASETS[3] = {
    "Housing.csv",
    "Student_Performance.csv",
    "multiple_linear_regression_dataset.csv"
};

/* ================== PHASE 2 ================== */
typedef enum { COL_NUMERIC, COL_CATEGORICAL } ColumnType;

char column_names[MAX_FEATURES_CONST][STRING_BUFFER_LIMIT_CONST];
ColumnType column_types[MAX_FEATURES_CONST];

double raw_numeric[MAX_SAMPLES_CONST][MAX_FEATURES_CONST];
char raw_categorical[MAX_SAMPLES_CONST][MAX_FEATURES_CONST][STRING_BUFFER_LIMIT_CONST];

int sample_count, feature_count, target_col_index;

/* ================== PHASE 3 ================== */
double X_norm[MAX_SAMPLES_CONST][MAX_EXPANDED_FEATURES_CONST];
double y_norm[MAX_SAMPLES_CONST];
int expanded_feature_count;

typedef struct {
    int original_col;
    int expanded_start;
    int expanded_width;
} FeatureMap;

FeatureMap feature_map[MAX_FEATURES_CONST];

/* ================== PHASE 4 ================== */
double beta[MAX_EXPANDED_FEATURES_CONST + 1];  /* +1 for bias */

/* ================== UTILS ================== */
void trim_whitespace(char *s) {
    char *e;
    while (*s == ' ' || *s == '\t') s++;
    e = s + strlen(s) - 1;
    while (e > s && (*e == '\n' || *e == '\r' || *e == ' ')) *e-- = 0;
}

/* ================== SERVER INIT ================== */
void init_server() {
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT_NUMBER);

    bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr));
    listen(server_fd, 5);
}

/* ================== CLIENT MENU ================== */
int client_menu(int fd) {
    char buf[64] = {0};
    const char *menu =
        "WELCOME TO PRICE PREDICTION SERVER\n\n"
        "1) Housing.csv\n"
        "2) Student_Performance.csv\n"
        "3) multiple_linear_regression_dataset.csv\n"
        "Choice: ";
    send(fd, menu, strlen(menu), 0);
    recv(fd, buf, sizeof(buf)-1, 0);
    int c = atoi(buf);
    return (c >= 1 && c <= 3) ? c-1 : -1;
}

/* ================== CSV PARSE ================== */
int is_double(const char *s, double *v) {
    char *e; errno = 0;
    double d = strtod(s, &e);
    if (errno || e == s || *e) return 0;
    if (v) *v = d;
    return 1;
}

void parse_csv(const char *f) {
    FILE *fp = fopen(f, "r");
    char line[4096];

    fgets(line, sizeof(line), fp);
    feature_count = 0;
    char *t = strtok(line, ",");
    while (t) {
        strcpy(column_names[feature_count], t);
        trim_whitespace(column_names[feature_count]);
        feature_count++;
        t = strtok(NULL, ",");
    }
    target_col_index = feature_count - 1;

    for (int i = 0; i < feature_count; i++)
        column_types[i] = COL_NUMERIC;

    sample_count = 0;
    while (fgets(line, sizeof(line), fp)) {
        char *cells[MAX_FEATURES_CONST];
        int c = 0;
        t = strtok(line, ",\n");
        while (t) {
            cells[c++] = t;
            t = strtok(NULL, ",\n");
        }
        if (c != feature_count) continue;

        for (int i = 0; i < feature_count; i++) {
            double tmp;
            if (column_types[i] == COL_NUMERIC &&
                !is_double(cells[i], &tmp))
                column_types[i] = COL_CATEGORICAL;
        }

        for (int i = 0; i < feature_count; i++) {
            if (column_types[i] == COL_NUMERIC)
                is_double(cells[i], &raw_numeric[sample_count][i]);
            else
                strcpy(raw_categorical[sample_count][i], cells[i]);
        }
        sample_count++;
    }
    fclose(fp);
}

/* ================== PHASE 3 ================== */
void run_preprocessing(int fd) {
    expanded_feature_count = 0;

    for (int c = 0; c < feature_count; c++) {
        feature_map[c].expanded_start = expanded_feature_count;
        if (column_types[c] == COL_NUMERIC) expanded_feature_count++;
        else if (!strcmp(column_names[c], "furnishingstatus")) expanded_feature_count += 3;
        else expanded_feature_count++;
    }

    for (int i = 0; i < sample_count; i++) {
        int col = 0;
        for (int c = 0; c < feature_count; c++) {
            if (column_types[c] == COL_NUMERIC) {
                X_norm[i][col++] = raw_numeric[i][c];
            } else {
                char *v = raw_categorical[i][c];
                if (!strcmp(column_names[c], "furnishingstatus")) {
                    X_norm[i][col++] = !strcmp(v, "furnished");
                    X_norm[i][col++] = !strcmp(v, "semi-furnished");
                    X_norm[i][col++] = !strcmp(v, "unfurnished");
                } else {
                    X_norm[i][col++] = (!strcmp(v, "yes"));
                }
            }
        }
        y_norm[i] = raw_numeric[i][target_col_index];
    }

    char msg[128];
    sprintf(msg,
        "\n[PHASE 3] Preprocessing complete\n"
        "Samples: %d\n"
        "Expanded features: %d\n\n",
        sample_count, expanded_feature_count);
    send(fd, msg, strlen(msg), 0);
}

/* ================== PHASE 4: TRAINING ================== */

typedef struct { int idx; } BetaArg;

void *beta_thread_NE(void *arg) {
    int j = ((BetaArg *)arg)->idx;
    double sum = 0;

    for (int i = 0; i < sample_count; i++) {
        double xj = (j == 0) ? 1.0 : X_norm[i][j-1];
        sum += xj * y_norm[i];
    }
    beta[j] = sum / sample_count;
    return NULL;
}

void *beta_thread_GD(void *arg) {
    int j = ((BetaArg *)arg)->idx;

    for (int it = 0; it < GD_ITERATIONS; it++) {
        double grad = 0;
        for (int i = 0; i < sample_count; i++) {
            double pred = beta[0];
            for (int k = 1; k <= expanded_feature_count; k++)
                pred += beta[k] * X_norm[i][k-1];
            double err = pred - y_norm[i];
            double xj = (j == 0) ? 1.0 : X_norm[i][j-1];
            grad += err * xj;
        }
        beta[j] -= GD_LEARNING_RATE * grad / sample_count;
    }
    return NULL;
}

void run_training(int fd) {
    char buf[64];
    const char *menu =
        "\nSelect training method:\n"
        "1) Normal Equation\n"
        "2) Gradient Descent\n"
        "Choice: ";
    send(fd, menu, strlen(menu), 0);
    recv(fd, buf, sizeof(buf)-1, 0);

    int method = atoi(buf);
    int total_beta = expanded_feature_count + 1;

    memset(beta, 0, sizeof(beta));

    pthread_t threads[MAX_EXPANDED_FEATURES_CONST];
    BetaArg args[MAX_EXPANDED_FEATURES_CONST];

    for (int j = 0; j < total_beta; j++) {
        args[j].idx = j;
        if (method == 1)
            pthread_create(&threads[j], NULL, beta_thread_NE, &args[j]);
        else
            pthread_create(&threads[j], NULL, beta_thread_GD, &args[j]);
    }

    for (int j = 0; j < total_beta; j++)
        pthread_join(threads[j], NULL);

    char out[2048];
    int off = 0;
    off += sprintf(out + off, "\n[PHASE 4] Training complete\n\n");
    for (int j = 0; j < total_beta; j++)
        off += sprintf(out + off, "beta_%d = %.6f\n", j, beta[j]);

    send(fd, out, strlen(out), 0);
}

/* ================== MAIN ================== */
int main() {
    init_server();
    pthread_mutex_init(&coeff_mutex, NULL);

    while (1) {
        int fd = accept(server_fd, NULL, NULL);
        if (fd < 0) continue;

        int idx = client_menu(fd);
        if (idx >= 0) {
            parse_csv(DATASETS[idx]);
            run_preprocessing(fd);
            run_training(fd);
        }
        close(fd);
    }
    return 0;
}
