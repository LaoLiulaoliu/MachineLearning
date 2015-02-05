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

#include "matrix.h"

#include <fstream>

using namespace matrix;

template < class T >
class Model
{
public:

    Model() {}
    virtual ~Model() {}

    int read_data();
    int print_X();
    int print_Y();


private:

    vector< vector<T> > X2d;
    vector< vector<T> > Y2d;
    Matrix< T,1,1 > X;
    Matrix< T,1,1 > Y;
};


template <class T>
int Model<T>::read_data()
{
    T x1, x2, y;
    ifstream infile("data.txt");
    while (infile >> x1 >> x2 >> y) {
        vector<T> vec_x(2);
        vec_x[0] = x1;
        vec_x[1] = x2;
        X2d.push_back(vec_x);

        vector<T> vec_y;
        vec_y.push_back(y);
        Y2d.push_back(vec_y);
    }

    return 0;
}

template <class T>
int Model<T>::print_X()
{
    for (typename vector< vector<T> >::iterator i = X2d.begin(); i != X2d.end(); ++i) {
        for (typename vector<T>::iterator j = i->begin(); j != i->end(); ++j) {
            cout << *j << " ";
        }
        cout << endl;
    }

    return 0;
}

template <class T>
int Model<T>::print_Y()
{
    for (size_t i = 0; i < Y2d.size(); ++i) {
        for (size_t j = 0; j < Y2d[i].size(); ++j) {
            cout << Y2d[i][j] << " ";
        }
        cout << endl;
    }
    return 0;
}

