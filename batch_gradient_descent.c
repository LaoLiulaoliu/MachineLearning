#include <stdio.h>

int main(void)
{
    int m = 4, n = 3, max_iteration = 5000;
    float loss = 1000.0; /* initialize a big loss */

    float matrix[4][3] = {{1, 1, 4}, {1, 2, 5}, {1, 5, 1}, {1, 4, 2}};
    float result[4] = {19, 26, 19, 20};
    float theta[3] = {1, 2, 3};     /* initialized theta {1, 2, 3}, model theta is {0, 3, 4} */
    float error_sum[3] = {0, 0, 0};
    float learning_rate = 0.0002; /* leaning_rate must be small enough */

    for(int iteration = 0; iteration < max_iteration && loss > 0.0001; ++iteration)
    {
        for(int i = 0; i < m; ++i) {
            float hypothesis = 0;
            for(int j = 0; j < n; ++j) {
                hypothesis += matrix[i][j] * theta[j];
            }
            for(int j = 0; j < n; ++j) {
                error_sum[j] += (result[i] - hypothesis) * matrix[i][j];
            }
        }
        for(int j = 0; j < n; ++j) {
            theta[j] += learning_rate * (error_sum[j]);
        }

        printf("\niteration: %d; theta: %f, %f, %f; ", iteration, theta[0], theta[1], theta[2]);

        loss = 0.0;
        for(int i = 0; i < m; ++i) {
            float sum = 0.0;
            for(int j = 0; j < n; ++j) {
                sum += matrix[i][j] * theta[j];
            }
            loss += (sum - result[i]) * (sum - result[i]);
        }
        printf("loss: %f\n", loss);
    }

    return 0;
}

