#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <math.h>

/* ================== CONSTANTS ================== */
#define PORT 60000
#define MAX_SAMPLES 10000
#define MAX_FEATURES 100
#define MAX_EXPANDED 256
#define STR_LEN 100

/* ================== GLOBAL DATA ================== */
int sample_count = 0;
int feature_count = 0;
int expanded_feature_count = 0;
int target_col_index = -1;

/* ================== DATA STORAGE ================== */
char column_names[MAX_FEATURES][STR_LEN];
double raw_numeric[MAX_SAMPLES][MAX_FEATURES];
char raw_categorical[MAX_SAMPLES][MAX_FEATURES][STR_LEN];
int is_numeric[MAX_FEATURES];

double X_norm[MAX_SAMPLES][MAX_EXPANDED];
double y_norm[MAX_SAMPLES];

/* Phase 4 */
double XtX[MAX_EXPANDED+1][MAX_EXPANDED+1];
double Xty[MAX_EXPANDED+1];
double beta[MAX_EXPANDED+1];

/* ================== DATASETS ================== */
const char *DATASETS[] = {
    "Housing.csv",
    "Student_Performance.csv",
    "multiple_linear_regression_dataset.csv"
};

/* ================== UTILS ================== */
int is_double(const char *s, double *out) {
    char *e;
    errno = 0;
    double v = strtod(s, &e);
    if (errno || e == s || *e != '\0') return 0;
    if (out) *out = v;
    return 1;
}

/* ================== CSV PARSER ================== */
void parse_csv(const char *fname) {
    FILE *fp = fopen(fname, "r");
    char line[4096];

    fgets(line, sizeof(line), fp);
    feature_count = 0;
    char *tok = strtok(line, ",\n");
    while (tok) {
        strcpy(column_names[feature_count++], tok);
        tok = strtok(NULL, ",\n");
    }
    target_col_index = feature_count - 1;

    for (int i = 0; i < feature_count; i++)
        is_numeric[i] = 1;

    sample_count = 0;
    while (fgets(line, sizeof(line), fp)) {
        char *cells[MAX_FEATURES];
        int c = 0;
        tok = strtok(line, ",\n");
        while (tok) {
            cells[c++] = tok;
            tok = strtok(NULL, ",\n");
        }
        if (c != feature_count) continue;

        for (int i = 0; i < feature_count; i++) {
            double v;
            if (is_numeric[i] && !is_double(cells[i], &v))
                is_numeric[i] = 0;
        }

        for (int i = 0; i < feature_count; i++) {
            if (is_numeric[i])
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
        if (c == target_col_index) continue;
        if (is_numeric[c]) expanded_feature_count++;
        else if (!strcmp(column_names[c], "furnishingstatus"))
            expanded_feature_count += 3;
        else expanded_feature_count++;
    }

    for (int i = 0; i < sample_count; i++) {
        int col = 0;
        for (int c = 0; c < feature_count; c++) {
            if (c == target_col_index) continue;
            if (is_numeric[c]) {
                X_norm[i][col++] = raw_numeric[i][c];
            } else {
                char *v = raw_categorical[i][c];
                if (!strcmp(column_names[c], "furnishingstatus")) {
                    X_norm[i][col++] = !strcmp(v, "furnished");
                    X_norm[i][col++] = !strcmp(v, "semi-furnished");
                    X_norm[i][col++] = !strcmp(v, "unfurnished");
                } else {
                    X_norm[i][col++] = !strcmp(v, "yes");
                }
            }
        }
        y_norm[i] = raw_numeric[i][target_col_index];
    }

    char msg[256];
    sprintf(msg,
        "\n[PHASE 3] Preprocessing complete\n"
        "Samples: %d\nExpanded features: %d\n\n",
        sample_count, expanded_feature_count);
    send(fd, msg, strlen(msg), 0);
}

/* ================== PHASE 4 THREAD ================== */
typedef struct { int j; } ThreadArg;

void *build_normal_eq_row(void *arg) {
    int j = ((ThreadArg*)arg)->j;

    for (int k = 0; k <= expanded_feature_count; k++)
        XtX[j][k] = 0.0;
    Xty[j] = 0.0;

    for (int i = 0; i < sample_count; i++) {
        double xj = (j == 0) ? 1.0 : X_norm[i][j-1];
        for (int k = 0; k <= expanded_feature_count; k++) {
            double xk = (k == 0) ? 1.0 : X_norm[i][k-1];
            XtX[j][k] += xj * xk;
        }
        Xty[j] += xj * y_norm[i];
    }
    return NULL;
}

/* ================== GAUSS–JORDAN ================== */
void solve_gauss_jordan(int n) {
    double aug[MAX_EXPANDED+1][MAX_EXPANDED+2];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            aug[i][j] = XtX[i][j];
        aug[i][n] = Xty[i];
    }

    for (int i = 0; i < n; i++) {
        double pivot = aug[i][i];
        if (fabs(pivot) < 1e-9) {
            for (int r = i+1; r < n; r++) {
                if (fabs(aug[r][i]) > 1e-9) {
                    for (int c = 0; c <= n; c++) {
                        double t = aug[i][c];
                        aug[i][c] = aug[r][c];
                        aug[r][c] = t;
                    }
                    pivot = aug[i][i];
                    break;
                }
            }
        }
        for (int c = 0; c <= n; c++)
            aug[i][c] /= pivot;

        for (int r = 0; r < n; r++) {
            if (r == i) continue;
            double f = aug[r][i];
            for (int c = 0; c <= n; c++)
                aug[r][c] -= f * aug[i][c];
        }
    }

    for (int i = 0; i < n; i++)
        beta[i] = aug[i][n];
}

/* ================== TRAINING ================== */
void run_training(int fd) {
    char buf[32];
    const char *menu =
        "\nSelect training method:\n"
        "1) Normal Equation\n"
        "2) Gradient Descent (not shown)\n"
        "Choice: ";
    send(fd, menu, strlen(menu), 0);
    recv(fd, buf, sizeof(buf)-1, 0);

    if (atoi(buf) == 1) {
        pthread_t threads[MAX_EXPANDED+1];
        ThreadArg args[MAX_EXPANDED+1];
        int total = expanded_feature_count + 1;

        for (int j = 0; j < total; j++) {
            args[j].j = j;
            pthread_create(&threads[j], NULL,
                           build_normal_eq_row, &args[j]);
        }
        for (int j = 0; j < total; j++)
            pthread_join(threads[j], NULL);

        solve_gauss_jordan(total);
    }

    char out[2048];
    int off = 0;
    off += sprintf(out+off,
        "\n[PHASE 4] Training complete\n\n"
        "y_norm = %.6f\n", beta[0]);
    for (int j = 1; j <= expanded_feature_count; j++)
        off += sprintf(out+off,
            " + %.6f * x%d_norm\n", beta[j], j);

    send(fd, out, off, 0);
}

/* ================== MAIN ================== */
int main() {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(PORT);

    bind(server_fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 5);

    while (1) {
        int fd = accept(server_fd, NULL, NULL);
        char buf[32];

        const char *menu =
            "WELCOME TO PRICE PREDICTION SERVER\n\n"
            "1) Housing.csv\n"
            "2) Student_Performance.csv\n"
            "3) multiple_linear_regression_dataset.csv\n"
            "Choice: ";
        send(fd, menu, strlen(menu), 0);
        recv(fd, buf, sizeof(buf)-1, 0);

        int choice = atoi(buf) - 1;
        if (choice >= 0 && choice < 3) {
            parse_csv(DATASETS[choice]);
            run_preprocessing(fd);
            run_training(fd);
        }
        close(fd);
    }
    return 0;
}
