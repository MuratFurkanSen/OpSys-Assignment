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
#define STR_LEN 100

/* ================== GLOBAL DATA ================== */
int sample_count = 0;
int feature_count = 0;
int expanded_feature_count = 0;
int target_col_index = -1;
char *END_OF_LINE = "\r\n";

/* ================== DATA STORAGE ================== */
char line_buffer[MAX_FEATURES*STR_LEN];
char *column_names[MAX_FEATURES];
char *raw_categorical[MAX_SAMPLES][MAX_FEATURES];
double raw_numeric[MAX_SAMPLES][MAX_FEATURES];
int is_numeric[MAX_FEATURES];

double X_norm[MAX_SAMPLES][MAX_FEATURES];
double X_norm_min[MAX_FEATURES];
double X_norm_max[MAX_FEATURES];

double y_norm[MAX_FEATURES];
double y_norm_min;
double y_norm_max;

double X_norm_transpose[MAX_FEATURES][MAX_SAMPLES];

double XT_X_norm[MAX_SAMPLES][MAX_SAMPLES];
double XT_y_norm[MAX_FEATURES];

double XT_X_inverse[MAX_SAMPLES][MAX_SAMPLES];
double beta[MAX_FEATURES];

double user_input[MAX_FEATURES];


/* ================== DATASETS ================== */
#define DATASET_COUNT 3

const char *DATASETS[] = {
    "Housing.csv",
    "Student_Performance.csv",
    "multiple_linear_regression_dataset.csv"
};



// Function Prototypes
int check_file_existence(void);
void parse_csv_file(FILE *fp);

int normalize_data();
int fill_intercepsts_column(int norm_col_index);
int normalize_numeric_column(int raw_col_index);
int normalize_categorical_column(int raw_col_index);
int normalize_target_column(int raw_col_index);

void transpose_matrix(double input[][MAX_FEATURES], double output[][MAX_SAMPLES], int rows, int cols);
void compute_XTX(double A[][MAX_SAMPLES], double B[][MAX_FEATURES], double C[][MAX_SAMPLES], int rowsA, int colsA, int colsB);
void compute_XTY(double X_T[][MAX_SAMPLES], double y[], double XTY[], int cols, int rows);
int invert_matrix(double A[][MAX_SAMPLES], double A_inv[][MAX_SAMPLES], int n);
void compute_beta(double XTX_inv[][MAX_SAMPLES], double XTY[], double beta[], int cols);

void ask_user_parameters();
double predict();
double denormalize_target(double normalized_val);

int is_double(const char *str);

int main(void) {
    const char *filename = DATASETS[2];

    // 1) Check file
    if (check_file_existence() == -1) {
        printf("Error: One or more dataset files are missing!\n");
        return EXIT_FAILURE;
    }

    // 2) Open file and parse CSV
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror(filename);
        return EXIT_FAILURE;
    }
    parse_csv_file(fp);
    fclose(fp);

    // 3) Normalize data (intercept + numeric + categorical + target)
    normalize_data();

    int rows = sample_count;
    int cols = feature_count;

    // 2) Transpose
    transpose_matrix(X_norm, X_norm_transpose, rows, cols);

    // 3) Multiply X_T * X
    compute_XTX(X_norm_transpose, X_norm, XT_X_norm, cols, rows, cols);


    compute_XTY(X_norm_transpose, y_norm, XT_y_norm, cols, rows);

    // 3) Invert XTX
    if (invert_matrix(XT_X_norm, XT_X_inverse, cols) != 0) {
        printf("Error: XTX is singular, cannot invert!\n");
        return -1;
    }

    // Assuming XTX_inv and XTY are already computed
    compute_beta(XT_X_inverse, XT_y_norm, beta, cols);

    // First one is intercept(bias)
    user_input[0] = 1;
    // Get the Rest of the Input
    ask_user_parameters();
    double prediction = predict();
    
    printf("Normalized Prediction Result %g\n", prediction);
    printf("Normalized Prediction Result %g\n", denormalize_target(prediction));

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
        // 2) Set intercept column as the first one
    fill_intercepsts_column(0);

    // 3) Normalize numeric columns (shifted by +1 in X_norm)
    for (int c = 0; c < feature_count; c++) {

        // Target Column
        if (c == feature_count -1){
            normalize_target_column(c);
            break;
        }

        // Feature Columns
        if (is_numeric[c] == 1){
            normalize_numeric_column(c);
        } else{
            normalize_categorical_column(c);
        }
    }
}

int fill_intercepsts_column(int norm_col_index){
    for (int r = 0; r < sample_count; r++) {
        X_norm[r][norm_col_index] = 1;
    }
    return 0;
}

int normalize_numeric_column(int raw_col_index){
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
    return 0;
}

int normalize_categorical_column(int raw_col_index){
    int norm_col_index = raw_col_index + 1;

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
    return 0;
}

int normalize_target_column(int raw_col_index){
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
    return 0;
}


// ================== MATRIX OPERATIONS FUNCTIONS ==================
void transpose_matrix(double input[][MAX_FEATURES], double output[][MAX_SAMPLES], int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            output[c][r] = input[r][c];
        }
    }
}

void compute_XTX(double A[][MAX_SAMPLES], double B[][MAX_FEATURES], double C[][MAX_SAMPLES], int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < colsA; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void compute_XTY(double X_T[][MAX_SAMPLES], double y[], double XTY[], int cols, int rows) {
    for (int i = 0; i < cols; i++) {
        XTY[i] = 0.0;
        for (int j = 0; j < rows; j++) {
            XTY[i] += X_T[i][j] * y[j];
        }
    }
}

int invert_matrix(double A[][MAX_SAMPLES], double A_inv[][MAX_SAMPLES], int n) {
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

void compute_beta(double XTX_inv[][MAX_SAMPLES], double XTY[], double beta[], int cols) {
    for (int i = 0; i < cols; i++) {
        beta[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            beta[i] += XTX_inv[i][j] * XTY[j];
        }
    }
}

// ================== PREDICTION FUNCTIONS ==================
void ask_user_parameters() {
    char input[STR_LEN];
    
    
    for (int c = 0; c < feature_count - 1; c++) { // Exclude target
        // 0 is Reserved for Intercept(Bias)
        int norm_col_index = c+1;
        while (1) {
            printf("Enter value for %s: ", column_names[c]);
            if (!fgets(input, sizeof(input), stdin)) {
                printf("Error reading input. Try again.\n");
                continue;
            }

            // Remove newline
            input[strcspn(input, END_OF_LINE)] = '\0';

            if (is_numeric[c] == 1) {
                char *endptr;
                double val = strtod(input, &endptr);
                if (*endptr != '\0' || input[0] == '\0') {
                    printf("Invalid numeric value. Please try again.\n");
                    continue;
                }
                double range = X_norm_max[norm_col_index]- X_norm_min[norm_col_index];
                if (range == 0){
                    range = 1;
                }
                user_input[norm_col_index] = (val-X_norm_min[norm_col_index])/range;
            } else {
                if (strlen(input) == 0) {
                    printf("Invalid categorical value. Please try again.\n");
                    continue;
                }

                if (strcasecmp(column_names[c], "furnishingstatus") == 0){
                    char *value = input;

                    if (strcasecmp(value, "furnished") == 0) {
                        user_input[norm_col_index]= 2;
                    } else if (strcasecmp(value, "semi-furnished") == 0) {
                        user_input[norm_col_index]= 1;
                    } else if (strcasecmp(value, "unfurnished") == 0) {
                        user_input[norm_col_index]= 0;
                    } else { // Unknown Value
                        user_input[norm_col_index]= -1;
                    }
                
                } else{
                    char *value = input;

                    if (strcasecmp(value, "yes") == 0) {
                        user_input[norm_col_index]= 1;
                    } else if (strcasecmp(value, "no") == 0) {
                        user_input[norm_col_index]= 0;
                    } else { // Unknown Value
                        user_input[norm_col_index]= -1;
                    }
                }
                if (user_input[norm_col_index] == -1){
                    printf("Invalid categorical value. Please try again.\n");
                    continue;   
                }
            }        
            break; // Valid input, go to next feature
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
    if (range == 0) range = 1;  // avoid division by zero
    return normalized_val * range + y_norm_min;
}

// ================== HELPER FUNCTIONS ==================

// Returns 1 when true, 0 otherwise.
int is_double(const char *str) {
    char *endptr;

    strtod(str, &endptr);

    return (*str != '\0' && *endptr == '\0');
}
