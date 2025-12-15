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
double y_norm[MAX_SAMPLES];



/* ================== DATASETS ================== */
#define DATASET_COUNT 3

const char *DATASETS[] = {
    "Housing.csv",
    "Student_Performance.csv",
    "multiple_linear_regression_dataset.csv"
};

int check_file_existence(void);

int is_double(const char *str);
void parse_csv_file(FILE *fp);

int main(void) {

    if (check_file_existence() != NULL){
        printf("File '%s' is missing", check_file_existence())
    }
    const char *filename = "multiple_linear_regression_dataset.csv";
    FILE *fp = fopen(filename, "r");

    if (!fp) {
        perror(filename);
        return EXIT_FAILURE;
    }

    parse_csv_file(fp);

    fclose(fp);

    // Print results
    printf("Columns: %d, Samples: %d\n", feature_count, sample_count);
    for (int i = 0; i < feature_count; i++) {
        printf("Column %d: %s (%s)\n", i, column_names[i], 
               is_numeric[i] ? "numeric" : "categorical");
    }

    for (int r = 0; r < sample_count; r++) {
        printf("Row %d: ", r);
        for (int c = 0; c < feature_count; c++) {
            if (is_numeric[c])
                printf("%g ", raw_numeric[r][c]);
            else
                printf("%s ", raw_categorical[r][c]);
        }
        printf("\n");
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
int normalize_numeric_column(int col_index){
    double min = raw_numeric[0][col_index];
    double max = raw_numeric[0][col_index];

    for (int r = 0; r < sample_count; r++) {
        if (raw_numeric[r][col_index] < min){
             min = raw_numeric[r][col_index];
        }
        if (raw_numeric[r][col_index] > max){
            max = raw_numeric[r][col_index];
        }
    }
    X_norm_min[col_index] = min;
    X_norm_max[col_index] = max;
    
    double range = max - min;
    if (range == 0){
        return -1;
    }
    for (int r = 0; r < sample_count; r++) {
        X_norm[r][col_index] = (raw_numeric[r][col_index] - min) / range;
    }   
    return 0;
}

void normalize_categorical_column(int col_index){
    if (strcasecmp(column_names[col_index], "furnishingstatus") == 0){
        for (int r = 0; r < sample_count; r++) {
            char *value = raw_categorical[r][col_index];

            if (strcasecmp(value, "furnished") == 0) {
                X_norm[r][col_index] = 2;
            } else if (strcasecmp(value, "semi-furnished") == 0) {
                X_norm[r][col_index] = 1;
            } else if (strcasecmp(value, "unfurnished") == 0) {
                X_norm[r][col_index] = 0;
            } else { // Unknown Value
                X_norm[r][col_index] = -1;
            }
        }
    } else{
        for (int r = 0; r < sample_count; r++) {
            char *value = raw_categorical[r][col_index];

            if (strcasecmp(value, "yes") == 0) {
                X_norm[r][col_index] = 1;
            } else if (strcasecmp(value, "no") == 0) {
                X_norm[r][col_index] = 0;
            } else { // Unknown Value
                X_norm[r][col_index] = -1;
            }
        }
    }
}

// ================== HELPER FUNCTIONS ==================
int check_file_existence(void) {
    for (int i = 0; i < DATASET_COUNT; i++) {
        if (!(access(DATASETS[i], F_OK) == 0)) {
          return -1;
        } 
    }
    return 0;
}

// Returns 1 when true, 0 otherwise.
int is_double(const char *str) {
    char *endptr;

    strtod(str, &endptr);

    return (*str != '\0' && *endptr == '\0');
}
