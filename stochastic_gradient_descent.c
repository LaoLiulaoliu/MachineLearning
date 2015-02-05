/*
 * MachineLearning, open source software machine learning.
 * Copyright (C) 2014-2015
 * mailto:miraclecome AT gmail DOT com
 *
 * MachineLearning is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * MachineLearning is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <stdio.h>

int main(void)
{
    int m = 4, n = 3, max_iteration = 50000;
    float loss = 1000.0; /* initialize a big loss */

    float matrix[4][3] = {{1, 1, 4}, {1, 2, 5}, {1, 5, 1}, {1, 4, 2}};
    float result[4] = {19, 26, 19, 20};
    float theta[3] = {1, 2, 3};     /* initialized theta {1, 2, 3}, model theta is {0, 3, 4} */
    float learning_rate = 0.01; /* leaning_rate must be small enough */

    for(int iteration = 0; iteration < max_iteration && loss > 0.0001; ++iteration)
    {
        float error_sum = 0;

        int i = iteration % 4;
        {
            learning_rate = 1.0 / (iteration + i + 100) + 0.0001;
            float hypothesis = 0;
            for(int j = 0; j < n; ++j) {
                hypothesis += matrix[i][j] * theta[j];
            }
            error_sum = result[i] - hypothesis;
            for(int j = 0; j < n; ++j) {
                theta[j] += learning_rate * error_sum * matrix[i][j];
            }
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

