#include "model.h"

int training(float matrix[][3],
             float *theta,
             float *result,
             float *error_sum,
             int m, int n, int max_iteration,
             float loss, float learning_rate)
{
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

int train_gradient_descent()
{
    int m = 4, n = 3, max_iteration = 5000;
    float loss = 1000.0; /* initialize a big loss */
    float learning_rate = 0.0002; /* leaning_rate must be small enough */

    float matrix[4][3] = {{1, 1, 4}, {1, 2, 5}, {1, 5, 1}, {1, 4, 2}};
    float theta[3] = {1, 2, 3};     /* initialized theta {1, 2, 3}, model theta is {0, 3, 4} */
    float result[4] = {19, 26, 19, 20};
    float error_sum[3] = {0, 0, 0};

    training(matrix, theta, result, error_sum, m, n, max_iteration, loss, learning_rate);
    return 0;
}

int main(void)
{
    Model<float> a;
    a.read_data();
    a.print_Y();
    return 0;
}

