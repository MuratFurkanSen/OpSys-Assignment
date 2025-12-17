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
#define STRING_BUFFER_LIMIT 100
#define PREPROC_THREAD_LIMIT 128
#define COEFF_THREAD_LIMIT 128


/* ================== GLOBAL DATA ================== */
int sample_count = 0;
int feature_count = 0;
int client_fd = -1;
char *END_OF_LINE = "\r\n";

/* ================== DATA STORAGE ================== */
char line_buffer[MAX_FEATURES*STRING_BUFFER_LIMIT];
char *column_names[MAX_FEATURES];
char *raw_categorical[MAX_SAMPLES][MAX_FEATURES];
double raw_numeric[MAX_SAMPLES][MAX_FEATURES];
int is_numeric[MAX_FEATURES];

double X_norm[MAX_SAMPLES][MAX_FEATURES];
double X_norm_min[MAX_FEATURES];
double X_norm_max[MAX_FEATURES];

double y_norm[MAX_SAMPLES];
double y_norm_min;
double y_norm_max;

double X_norm_transpose[MAX_FEATURES][MAX_SAMPLES];

double XT_X_norm[MAX_FEATURES][MAX_FEATURES];
double XT_y_norm[MAX_FEATURES];

double XT_X_inverse[MAX_FEATURES][MAX_FEATURES];
double beta[MAX_FEATURES];

double user_input[MAX_FEATURES];

/* ================== DATASETS ================== */
#define DATASET_COUNT 3

const char *DATASETS[] = {
    "Housing.csv",
    "Student_Performance.csv",
    "multiple_linear_regression_dataset.csv"
};

/* ================== THREAD ARGUMENT STRUCTS ================== */
typedef struct {
    int row;
    int colsA;
    int colsB;
    double (*A)[MAX_SAMPLES];
    double (*B)[MAX_FEATURES];
    double (*C)[MAX_FEATURES];
} matmul_thread_arg;

typedef struct {
    int index;
    int rows;
    double (*X_T)[MAX_SAMPLES];
    double *y;
    double *XTY;
} xty_thread_arg;






// Function Prototypes
int check_file_existence(void);
void parse_csv_file(FILE *fp);

int normalize_data();
void *fill_intercepsts_column(void *arg);
void *normalize_numeric_column(void *arg);
void *normalize_categorical_column(void *arg);
void *normalize_target_column(void *arg);

void transpose_matrix(double input[][MAX_FEATURES], double output[][MAX_SAMPLES], int rows, int cols);
void compute_XTX_threaded(double A[][MAX_SAMPLES], double B[][MAX_FEATURES], double C[][MAX_FEATURES], int rowsA, int colsA, int colsB);
void *compute_XTX_row(void *arg);
void compute_XTY_threaded(double X_T[][MAX_SAMPLES], double y[MAX_SAMPLES], double XTY[MAX_FEATURES], int cols, int rows);
void *compute_XTY_element(void *arg);
int invert_matrix(double A[][MAX_FEATURES], double A_inv[][MAX_FEATURES], int n);
void compute_beta(double XTX_inv[][MAX_FEATURES], double XTY[MAX_FEATURES], double beta[], int cols);

void ask_user_parameters_client();
double predict();
double denormalize_target(double normalized_val);

int setup_server_and_accept(void);
int ask_csv_filename_server(char *filename, size_t size);
int send_to_client(const char *msg);
void send_dataset_summary(const char *filename);
void send_beta_equation_to_client();

int is_double(const char *str);
void free_allocated_memory(void);

int main(void) {
    client_fd = setup_server_and_accept();
    send_to_client("WELCOME TO PRICE PREDICTION SERVER\n\n");

    // 1) Check all 3 CSV files exist
    if (check_file_existence() == -1) {
        send_to_client("Error: One or more dataset files are missing!\n");
        close(client_fd);
        return EXIT_FAILURE;
    }

    // 2) Ask for CSV filename from client
    char filename[STRING_BUFFER_LIMIT];

    if (ask_csv_filename_server(filename, sizeof(filename)) != 0) {
        send_to_client("Error receiving filename from client.\n");
        close(client_fd);
        return EXIT_FAILURE;
    }

    // 3) Open file and parse CSV
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        send_to_client("Error: Could not open specified file.\n");
        close(client_fd);
        return EXIT_FAILURE;
    }

    parse_csv_file(fp);
    fclose(fp);
    // 5) Send dataset summary to client
    send_dataset_summary(filename);

    // 6) Normalize data (intercept + numeric + categorical + target)
    normalize_data();

    // 7) Design MAtrix Creation Handled During Normalization
    send_to_client("Normalized feature matrix X_norm has been created.\n");
    send_to_client("Normalized target vector vector y_norm has been created.\n\n");


    // For Matrix Operations
    int rows = sample_count;
    int cols = feature_count;

    // 8) Take Transpose of Design Matrix X
    transpose_matrix(X_norm, X_norm_transpose, rows, cols);

    // 9) Multiply X_T * X
    send_to_client("Spawning coefficient calculation threads...\n");
    compute_XTX_threaded(X_norm_transpose, X_norm, XT_X_norm, cols, rows, cols);

    // 10) Multiply X_T * y
    compute_XTY_threaded(X_norm_transpose, y_norm, XT_y_norm, cols, rows);

    // 11) Invert XT_X for Solving Beras
    send_to_client("Solving (XᵀX)β = Xᵀy ...\n");
    if (invert_matrix(XT_X_norm, XT_X_inverse, cols) != 0) {
        send_to_client("Error: XTX is singular, cannot invert!\n");
        return -1;
    }
    
    // 12) Compute Beta Coefficients using β = (XᵀX)^(-1) * Xᵀy
    compute_beta(XT_X_inverse, XT_y_norm, beta, cols);
    send_to_client("Training completed.\n");

    send_to_client("FINAL MODEL (Normalized Form):\n\n");

    // 13) Send Beta Equation to Client
    send_beta_equation_to_client();
    
    // Prediction Loop
    while (1) {
        // Prompt for prediction
        send_to_client("\n");
        send_to_client("You can now input feature values to get a price prediction.\n\n");

        // First one is intercept(bias)
        user_input[0] = 1;

        // Get the Rest of the Input
        ask_user_parameters_client();
        double prediction = predict();
        
        // Send Prediction Results to Client
        send_to_client("\nPREDICTION RESULTS:\n");
        char buffer[128];

        // Normalized prediction
        snprintf(buffer, sizeof(buffer), "Normalized Prediction Result %.4f\n", prediction);
        send_to_client(buffer);

        // Real prediction (denormalized)
        snprintf(buffer, sizeof(buffer), "Real Prediction Result %.4f\n\n", denormalize_target(prediction));
        send_to_client(buffer);

        // Ask client if they want to repeat
        char response[STRING_BUFFER_LIMIT];
        send_to_client("Do you want to make another prediction? (yes/no):\n");
        int n = recv(client_fd, response, sizeof(response) - 1, 0);

        // Handle recv errors or disconnections
        if (n <= 0) {
            send_to_client("Error reading input. Exiting.\n");
            break;
        }
        response[n] = '\0';
        response[strcspn(response, END_OF_LINE)] = '\0';

        // If Client Decides to Quit Break Loop
        if (strcasecmp(response, "yes") != 0) {
            break;
        }
    }

    // Goodbye Message
    send_to_client("Thank you for using PRICE PREDICTION SERVER! Good Bye!\n\n");

    // Cleanup
    close(client_fd);
    free_allocated_memory(); 
    return 0;
}

// ================== FILE OPERATIONS FUNCTIONS ==================

int check_file_existence(void) {
    for (int i = 0; i < DATASET_COUNT; i++) {
        if (!(access(DATASETS[i], F_OK) == 0)) {
          return -1;
        } 
    }
    return 0;
}

void parse_csv_file(FILE *fp) {
    // Local Varaibles
    char *token;
    int col_counter = 0;
    int row_counter = 0;

    // Fill Header Names and Count Feature Size
    fgets(line_buffer, sizeof(line_buffer), fp);
    line_buffer[strcspn(line_buffer, END_OF_LINE)] = '\0';

    token = strtok(line_buffer, ",");
    while (token != NULL){
        column_names[col_counter] = strdup(token);
        token = strtok(NULL, ",");
        col_counter++;
        feature_count++;
    }
    col_counter = 0;

    // Fill First Row and Also Determine Column Types
    fgets(line_buffer, sizeof(line_buffer), fp);
    line_buffer[strcspn(line_buffer, END_OF_LINE)] = '\0';


    token = strtok(line_buffer, ",");
    while (token != NULL){
        if (is_double(token) == 1){
            is_numeric[col_counter] = 1;
            raw_numeric[sample_count][col_counter] = strtod(token, NULL);
        } else{
            is_numeric[col_counter] = 0;
            raw_categorical[sample_count][col_counter] = strdup(token);
        }
        token = strtok(NULL, ",");
        col_counter++;
    }
    sample_count++;
    col_counter = 0;

    
    // Fill Rest of the Columns
    while (fgets(line_buffer, sizeof(line_buffer), fp)) {
        line_buffer[strcspn(line_buffer, END_OF_LINE)] = '\0';
        token = strtok(line_buffer,",");

        while (token != NULL){
            if (is_numeric[col_counter] == 1){
                raw_numeric[sample_count][col_counter] = strtod(token, NULL);
            }else{
                raw_categorical[sample_count][col_counter] = strdup(token);
            }
            token = strtok(NULL, ",");
            col_counter++;
         }
        sample_count++;
        col_counter = 0;
    }
}
// ================== NORMALIZER FUNCTIONS ==================
int normalize_data(){
    // A thread per (Intercept + features + target)
    int total_thread_count = feature_count+1;

    int thread_index = 0;
    pthread_t threads[total_thread_count];
    int thread_col_indexes[total_thread_count];

    // Spawn intercept thread and fill intercept column as the first one
    thread_col_indexes[thread_index] = 0;
    pthread_create(&threads[thread_index], NULL, fill_intercepsts_column, &thread_col_indexes[thread_index]);
    thread_index++;

    // Normalize columns (shifted by +1 in X_norm)
    for (int c = 0; c < feature_count; c++) {
        thread_col_indexes[thread_index] = c;
        // Target Column
        if (c == feature_count -1){
            pthread_create(&threads[thread_index], NULL, normalize_target_column, &thread_col_indexes[thread_index]);
            thread_index++;
            break;
        }

        // Feature Columns
        if (is_numeric[c] == 1){
            pthread_create(&threads[thread_index], NULL, normalize_numeric_column, &thread_col_indexes[thread_index]);
            thread_index++;
        } else{ // Numeric Column
            pthread_create(&threads[thread_index], NULL, normalize_categorical_column, &thread_col_indexes[thread_index]);
            thread_index++;
        }
    }
    // Wait for Threads to Finish
    for (int i = 0; i < total_thread_count; i++) {
        pthread_join(threads[i], NULL);
    }
    send_to_client("\n");
    send_to_client("[OK] All normalization threads completed.\n\n");
    return 0;
}

void *fill_intercepsts_column(void *arg){
    int raw_col_index = *(int *)arg;
    for (int r = 0; r < sample_count; r++) {
        X_norm[r][raw_col_index] = 1;
    }
    return NULL;
}

void *normalize_numeric_column(void *arg){
    int raw_col_index = *(int *)arg;
    int norm_col_index = raw_col_index + 1;

    double min = raw_numeric[0][raw_col_index];
    double max = raw_numeric[0][raw_col_index];

    for (int r = 0; r < sample_count; r++) {
        if (raw_numeric[r][raw_col_index] < min){
             min = raw_numeric[r][raw_col_index];
        }
        if (raw_numeric[r][raw_col_index] > max){
            max = raw_numeric[r][raw_col_index];
        }
    }
    X_norm_min[norm_col_index] = min;
    X_norm_max[norm_col_index] = max;
    
    double range = max - min;
    if (range == 0){
        range = 1;
    }
    for (int r = 0; r < sample_count; r++) {
        X_norm[r][norm_col_index] = (raw_numeric[r][raw_col_index] - min) / range;
    }
    /* ---- THREAD INFO TO CLIENT ---- */
    
    char buffer[256];
    snprintf(buffer, sizeof(buffer),
                "[Thread N%d] Normalizing %s... xmin=%.0f xmax=%.0f\n",
                raw_col_index + 1,
                column_names[raw_col_index],
                min,
                max);
    send_to_client(buffer);
    
    /* ------------------------------------------------ */
    return NULL;
}

void *normalize_categorical_column(void *arg){
    int raw_col_index = *(int *)arg;
    int norm_col_index = raw_col_index + 1;
    /* ---- THREAD INFO TO CLIENT ---- */

    char buffer[256];

    if (strcasecmp(column_names[raw_col_index], "furnishingstatus") == 0) {
        snprintf(buffer, sizeof(buffer),
                 "[Thread C%d] %s: furnished→2, semi-furnished→1, unfurnished→0\n",
                 raw_col_index + 1,
                 column_names[raw_col_index]);
    } else {
        snprintf(buffer, sizeof(buffer),
                 "[Thread C%d] %s: yes/no → 1/0\n",
                 raw_col_index + 1,
                 column_names[raw_col_index]);
    }

    send_to_client(buffer);

    /* ------------------------------------------------ */


    if (strcasecmp(column_names[raw_col_index], "furnishingstatus") == 0){
        for (int r = 0; r < sample_count; r++) {
            char *value = raw_categorical[r][raw_col_index];

            if (strcasecmp(value, "furnished") == 0) {
                X_norm[r][norm_col_index] = 2;
            } else if (strcasecmp(value, "semi-furnished") == 0) {
                X_norm[r][norm_col_index] = 1;
            } else if (strcasecmp(value, "unfurnished") == 0) {
                X_norm[r][norm_col_index] = 0;
            } else { // Unknown Value
                X_norm[r][norm_col_index] = -1;
            }
        }
    } else{
        for (int r = 0; r < sample_count; r++) {
            char *value = raw_categorical[r][raw_col_index];

            if (strcasecmp(value, "yes") == 0) {
                X_norm[r][norm_col_index] = 1;
            } else if (strcasecmp(value, "no") == 0) {
                X_norm[r][norm_col_index] = 0;
            } else { // Unknown Value
                X_norm[r][norm_col_index] = -1;
            }
        }
    }
    return NULL;
}

void *normalize_target_column(void *arg){
    int raw_col_index = *(int *)arg;
    int norm_col_index = raw_col_index + 1;

    double min = raw_numeric[0][raw_col_index];
    double max = raw_numeric[0][raw_col_index];

    for (int r = 0; r < sample_count; r++) {
        if (raw_numeric[r][raw_col_index] < min){
             min = raw_numeric[r][raw_col_index];
        }
        if (raw_numeric[r][raw_col_index] > max){
            max = raw_numeric[r][raw_col_index];
        }
    }
    y_norm_min = min;
    y_norm_max = max;
    
    double range = max - min;
    if (range == 0){
        range = 1;
    }
    for (int r = 0; r < sample_count; r++) {
        y_norm[r] = (raw_numeric[r][raw_col_index] - min) / range;
    }

    /* ---- THREAD INFO TO CLIENT ---- */
    
    char buffer[256];
    snprintf(buffer, sizeof(buffer),
                "[Thread N%d] Normalizing %s (TARGET)... ymin=%.0f ymax=%.0f\n",
                raw_col_index + 1,
                column_names[raw_col_index],
                min,
                max);
    send_to_client(buffer);
    
    /* ------------------------------------------------ */
    return NULL;
}


// ================== MATRIX OPERATIONS FUNCTIONS ==================
void transpose_matrix(double input[][MAX_FEATURES], double output[][MAX_SAMPLES], int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            output[c][r] = input[r][c];
        }
    }
}

void compute_XTX_threaded( double A[][MAX_SAMPLES], double B[][MAX_FEATURES],double C[][MAX_FEATURES], int rowsA, int colsA, int colsB) {
    if (rowsA > COEFF_THREAD_LIMIT) {
        fprintf(stderr, "Too many threads\n");
        exit(1);
    }

    pthread_t threads[rowsA];
    matmul_thread_arg args[rowsA];

    for (int i = 0; i < rowsA; i++) {
        args[i].row = i;
        args[i].colsA = colsA;
        args[i].colsB = colsB;
        args[i].A = A;
        args[i].B = B;
        args[i].C = C;

        pthread_create(
            &threads[i],
            NULL,
            compute_XTX_row,
            &args[i]
        );
    }

    for (int i = 0; i < rowsA; i++) {
        pthread_join(threads[i], NULL);
    }
    send_to_client("\n");
    send_to_client("All coefficient calculation threads joined.\n\n");
}

void *compute_XTX_row(void *arg) {
    matmul_thread_arg *data = (matmul_thread_arg *)arg;
    int i = data->row;


    /* ---- THREAD INFO TO CLIENT ---- */
    char buffer[256];

    if (i == 0) {
        snprintf(buffer, sizeof(buffer),
                 "[β-Thread %d] Preparing data for β%d (intercept)...\n",
                 i + 1, i);
    } else {
        snprintf(buffer, sizeof(buffer),
                 "[β-Thread %d] Preparing data for β%d (%s)...\n",
                 i + 1, i,
                 column_names[i - 1]);
    }

    send_to_client(buffer);
    /* ----------------------------------------------------- */

    for (int j = 0; j < data->colsB; j++) {
        data->C[i][j] = 0.0;
        for (int k = 0; k < data->colsA; k++) {
            data->C[i][j] += data->A[i][k] * data->B[k][j];
        }
    }

    return NULL;
}

void compute_XTY_threaded( double X_T[][MAX_SAMPLES], double y[MAX_SAMPLES], double XTY[MAX_FEATURES], int cols, int rows) {
    pthread_t threads[cols];
    xty_thread_arg args[cols];

    for (int i = 0; i < cols; i++) {
        args[i].index = i;
        args[i].rows = rows;
        args[i].X_T = X_T;
        args[i].y = y;
        args[i].XTY = XTY;

        pthread_create(
            &threads[i],
            NULL,
            compute_XTY_element,
            &args[i]
        );
    }

    for (int i = 0; i < cols; i++) {
        pthread_join(threads[i], NULL);
    }
}

void *compute_XTY_element(void *arg) {
    xty_thread_arg *data = (xty_thread_arg *)arg;
    int i = data->index;

    data->XTY[i] = 0.0;
    for (int j = 0; j < data->rows; j++) {
        data->XTY[i] += data->X_T[i][j] * data->y[j];
    }

    return NULL;
}


int invert_matrix(double A[][MAX_FEATURES], double A_inv[][MAX_FEATURES], int n) {
    // Initialize A_inv as identity matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_inv[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Make a copy of A to work on
    double temp[MAX_FEATURES][MAX_FEATURES];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            temp[i][j] = A[i][j];

    // Gauss-Jordan elimination
    for (int i = 0; i < n; i++) {
        // Find pivot
        if (temp[i][i] == 0.0) {
            // Try to swap with a lower row
            int swap_row = -1;
            for (int k = i + 1; k < n; k++) {
                if (temp[k][i] != 0.0) {
                    swap_row = k;
                    break;
                }
            }
            if (swap_row == -1) return -1; // Singular matrix
            // Swap rows
            for (int j = 0; j < n; j++) {
                double t = temp[i][j]; temp[i][j] = temp[swap_row][j]; temp[swap_row][j] = t;
                t = A_inv[i][j]; A_inv[i][j] = A_inv[swap_row][j]; A_inv[swap_row][j] = t;
            }
        }

        // Normalize pivot row
        double pivot = temp[i][i];
        for (int j = 0; j < n; j++) {
            temp[i][j] /= pivot;
            A_inv[i][j] /= pivot;
        }

        // Eliminate other rows
        for (int k = 0; k < n; k++) {
            if (k == i) continue;
            double factor = temp[k][i];
            for (int j = 0; j < n; j++) {
                temp[k][j] -= factor * temp[i][j];
                A_inv[k][j] -= factor * A_inv[i][j];
            }
        }
    }

    return 0; // Success
}

void compute_beta(double XTX_inv[][MAX_FEATURES], double XTY[MAX_FEATURES], double beta[], int cols) {
    for (int i = 0; i < cols; i++) {
        beta[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            beta[i] += XTX_inv[i][j] * XTY[j];
        }
    }
}

// ================== PREDICTION FUNCTIONS ==================
void ask_user_parameters_client() {
    char buffer[STRING_BUFFER_LIMIT];

    for (int c = 0; c < feature_count - 1; c++) { // Exclude target
        int norm_col_index = c + 1;

        while (1) {
            // Send prompt to client
            snprintf(buffer, sizeof(buffer), "Enter value for %s: ", column_names[c]);
            send_to_client(buffer);

            // Receive input from client
            int n = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
            if (n <= 0) {
                send_to_client("Error reading input. Try again.\n");
                continue;
            }
            buffer[n] = '\0';

            // Remove newline or carriage return
            buffer[strcspn(buffer, END_OF_LINE)] = '\0';

            // Handle numeric input
            if (is_numeric[c] == 1) {
                char *endptr;
                double val = strtod(buffer, &endptr);
                if (*endptr != '\0' || buffer[0] == '\0') {
                    send_to_client("Invalid numeric value. Please try again.\n");
                    continue;
                }
                double range = X_norm_max[norm_col_index] - X_norm_min[norm_col_index];
                if (range == 0) range = 1;
                user_input[norm_col_index] = (val - X_norm_min[norm_col_index]) / range;

            } else { // Handle categorical input
                if (strlen(buffer) == 0) {
                    send_to_client("Invalid categorical value. Please try again.\n");
                    continue;
                }

                if (strcasecmp(column_names[c], "furnishingstatus") == 0) {
                    if (strcasecmp(buffer, "furnished") == 0) user_input[norm_col_index] = 2;
                    else if (strcasecmp(buffer, "semi-furnished") == 0) user_input[norm_col_index] = 1;
                    else if (strcasecmp(buffer, "unfurnished") == 0) user_input[norm_col_index] = 0;
                    else user_input[norm_col_index] = -1;

                } else {
                    if (strcasecmp(buffer, "yes") == 0) user_input[norm_col_index] = 1;
                    else if (strcasecmp(buffer, "no") == 0) user_input[norm_col_index] = 0;
                    else user_input[norm_col_index] = -1;
                }

                if (user_input[norm_col_index] == -1) {
                    send_to_client("Invalid categorical value. Please try again.\n");
                    continue;
                }
            }

            break; // valid input, move to next feature
        }
    }
}

double predict() {
    double result = 0.0;
    for (int i = 0; i < feature_count; i++) {
        result += beta[i] * user_input[i];
    }
    return result;
}

double denormalize_target(double normalized_val) {
    double range = y_norm_max - y_norm_min;
    if (range == 0){  // avoid division by zero
        range = 1;
    }  
    return normalized_val * range + y_norm_min;
}

// ================== SERVER FUNCTIONS ==================
int setup_server_and_accept(void) {
    int server_fd, client_fd;
    struct sockaddr_in server_addr;
    socklen_t addr_len = sizeof(server_addr);

    // Create socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    // Allow quick reuse
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // Setup address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    // Bind
    if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    // Listen
    if (listen(server_fd, 1) < 0) {
        perror("listen");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    printf("Server listening on port %d...\n", PORT);

    // Accept client
    client_fd = accept(server_fd, (struct sockaddr *)&server_addr, &addr_len);
    if (client_fd < 0) {
        perror("accept");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    return client_fd; // ready to send()
}

int ask_csv_filename_server(char *filename, size_t size) {
    const char *prompt = "Enter CSV file name to load:\n";

    // Send prompt
    if (send(client_fd, prompt, strlen(prompt), 0) <= 0) {
        return -1;
    }

    // Receive input
    ssize_t bytes = recv(client_fd, filename, size - 1, 0);
    if (bytes <= 0) {
        return -1;
    }

    filename[bytes] = '\0';

    // Strip \r\n
    filename[strcspn(filename, "\r\n")] = '\0';

    return 0;
}

int send_to_client(const char *msg) {
    size_t len = strlen(msg);
    ssize_t sent = 0;

    while (sent < len) {
        ssize_t n = send(client_fd, msg + sent, len - sent, 0);
        if (n <= 0) {
            return -1;  // error or client disconnected
        }
        sent += n;
    }

    return 0; // success
}

void send_dataset_summary( const char *filename) {
    char buffer[1024];

    // Header
    snprintf(buffer, sizeof(buffer),
             "\n[OK] File \"%s\" found.\n\n"
             "Reading file...\n"
             "%d rows loaded.\n"
             "%d columns detected.\n"
             "Column analysis:\n\n",
             filename, sample_count, feature_count);
    send_to_client(buffer);

    // Column details
    for (int c = 0; c < feature_count; c++) {
        if (is_numeric[c]) {
            if (c == feature_count - 1) {
                // Target column
                snprintf(buffer, sizeof(buffer),
                         "%s : numeric (TARGET)\n",
                         column_names[c]);
            } else {
                snprintf(buffer, sizeof(buffer),
                         "%s : numeric\n",
                         column_names[c]);
            }
        } else {
            if (strcasecmp(column_names[c], "furnishingstatus") == 0) {
                snprintf(buffer, sizeof(buffer),
                         "%s : categorical (furnished, semi-furnished, unfurnished)\n",
                         column_names[c]);
            } else {
                snprintf(buffer, sizeof(buffer),
                         "%s : categorical (yes/no)\n",
                         column_names[c]);
            }
        }
        send_to_client(buffer);
        
    }
    send_to_client("\n");
}

void send_beta_equation_to_client() {
    char buffer[512];

    // Intercept
    snprintf(buffer, sizeof(buffer), "price_norm = %.4f\n", beta[0]);
    send_to_client(buffer);

    // Rest of the features
    for (int i = 1; i < feature_count; i++) {
        // For formatting, prepend '+' for all except the first line handled above
        snprintf(buffer, sizeof(buffer), "+ %.4f * %s\n", beta[i], column_names[i - 1]);
        send_to_client(buffer);
    }
}

// ================== HELPER FUNCTIONS ==================

// Returns 1 when true, 0 otherwise.
int is_double(const char *str) {
    char *endptr;

    strtod(str, &endptr);

    return (*str != '\0' && *endptr == '\0');
}

void free_allocated_memory(void) {
    // Free column names
    for (int c = 0; c < feature_count; c++) {
        if (column_names[c] != NULL) {
            free(column_names[c]);
            column_names[c] = NULL;
        }
    }

    // Free categorical raw data
    for (int r = 0; r < sample_count; r++) {
        for (int c = 0; c < feature_count; c++) {
            if (is_numeric[c] == 0 && raw_categorical[r][c] != NULL) {
                free(raw_categorical[r][c]);
                raw_categorical[r][c] = NULL;
            }
        }
    }
}
