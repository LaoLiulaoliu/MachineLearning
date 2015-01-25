#include <iostream>
#include <fstream>
#include <vector>

#include "EMatrix.h"

using namespace ematrix;
using namespace std;

template <class T>
class Model
{
public:
    int read_data();
    int print_X();
    int print_Y();

private:
    vector< vector<T> > X2d;
    vector<T> Y2d;
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
        Y2d.push_back(y);
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
    size_t j = Y2d.size();
    for (j = 0; j < Y2d.size(); ++j) {
        cout << Y2d[j] << endl;
    }
    return 0;
}

template <class T>
int Model<T> operator * (const Model<T>& R)
{

}

