#ifndef __GBM_HPP_
#define __GBM_HPP_

#include <cstdlib>
#include <vector>
#include <random>

class GBMParam
{
private:
    double p;
    double m;
    double s;
    double d;
public:
    GBMParam() {}
    GBMParam(double p0, double mu, double sigma) {
        p = p0;
        m = mu;
        s = sigma;
        d = m + 0.5 * pow(s, 2);
    }
    ~GBMParam() {}

    double p0();
    double mu();
    double sigma();
    double drift();
};

void normal(std::vector<std::vector<double>> &dat, std::default_random_engine &seed);
void cumsum(std::vector<std::vector<double>> &dat);

std::vector<std::vector<double>> gbm(std::vector<GBMParam> &param, unsigned int ext, std::default_random_engine &seed);

#endif