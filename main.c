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
char *END_OF_FILE = "\r\n";

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
double y_norm_min[MAX_FEATURES];
double y_norm_max[MAX_FEATURES];

double X_norm_transform[MAX_FEATURES][MAX_SAMPLES];



/* ================== DATASETS ================== */
#define DATASET_COUNT 3

const char *DATASETS[] = {
    "Housing.csv",
    "Student_Performance.csv",
    "multiple_linear_regression_dataset.csv"
};



// Main Function Prototypes
int check_file_existence(void);
void parse_csv_file(FILE *fp);

int normalize_data();
int fill_intercepsts_column(int norm_col_index);
int normalize_numeric_column(int raw_col_index);
int normalize_categorical_column(int raw_col_index);
int normalize_target_column(int raw_col_index);

void transpose_matrix(double input[][MAX_FEATURES], double output[][MAX_SAMPLES], int rows, int cols);

// Helper Function Prototypes
int is_double(const char *str);


#define ROWS 3
#define COLS 2
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

    // 4) Transpose the design matrix X_norm
    transpose_matrix(X_norm, X_norm_transform, sample_count, feature_count); 
    // +1 because first column is intercept

    // 5) Print X_norm and X_norm_transform to verify transpose
    printf("X_norm (original, %d x %d):\n", sample_count, feature_count);
    for (int r = 0; r < sample_count; r++) {
        for (int c = 0; c < feature_count + 1; c++) {
            printf("%g ", X_norm[r][c]);
        }
        printf("\n");
    }

    printf("\nX_norm_transform (transposed, %d x %d):\n", feature_count, sample_count);
    for (int r = 0; r < feature_count; r++) {
        for (int c = 0; c < sample_count; c++) {
            printf("%g - ", X_norm_transform[r][c]);
        }
        printf("\n");
    }

    // 6) Print target vector
    printf("\nY_norm (target):\n");
    for (int r = 0; r < sample_count; r++) {
        printf("%g\n", y_norm[r]);
    }

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
    line_buffer[strcspn(line_buffer, END_OF_FILE)] = '\0';

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
    line_buffer[strcspn(line_buffer, END_OF_FILE)] = '\0';


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
        line_buffer[strcspn(line_buffer, END_OF_FILE)] = '\0';
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
    y_norm_min[norm_col_index] = min;
    y_norm_max[norm_col_index] = max;
    
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

// ================== HELPER FUNCTIONS ==================

// Returns 1 when true, 0 otherwise.
int is_double(const char *str) {
    char *endptr;

    strtod(str, &endptr);

    return (*str != '\0' && *endptr == '\0');
}
