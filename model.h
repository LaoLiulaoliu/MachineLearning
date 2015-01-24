#include <iostream>
#include <fstream>
#include <vector>

#include "EMatrix.h"

using namespace ematrix;
using namespace std;

class Model
{
public:
    int read_data()
    {
        float x1, x2, y;
        ifstream infile("data.txt");
        cout << "right:";
        while (infile >> x1 >> x2 >> y) {
            vector<float> vec_x(2);
            vec_x[0] = x1;
            vec_x[1] = x2;
            X2d.push_back(vec_x);
            Y2d.push_back(y);
        }

        return 0;
    }

    int print_data()
    {
        vector< vector<float> >::iterator i;
        for (i = X2d.begin(); i != X2d.end(); ++i) {
            for (vector<float>::iterator j = i->begin(); j != i->end(); ++j) {
                cout << *j << " ";
            }
            cout << endl;
        }
        for (vector<float>::iterator j = Y2d.begin(); j != Y2d.end(); ++j) {
            cout << *j << endl;
        }
        return 0;
    }

private:
    vector< vector<float> > X2d;
    vector<float> Y2d;
};

