#include <cstdlib>
#include <vector>
#include <random>
#include <fstream>

#include "../lib/net.hpp"

double relu(double x) { return std::max(0.00, x); }
double drelu(double x) { return x < 0.00 ? 0.00 : 1.00; }

double Node::bias() { return b; }
double Node::sum() { return s; }
double Node::act() { return z; }
double Node::err() { return e; }
double Node::weight(unsigned int index) { return w[index]; }

void Node::init() { s = z = e = 0.00; }
void Node::set_bias(double val) { b = val; }
void Node::set_sum(double val) { s = val; }
void Node::set_act(double val) { z = val; }
void Node::add_err(double val) { e += val; }
void Node::set_weight(unsigned int index, double val) { w[index] = val; }

unsigned int Layer::in_features() { return in; }
unsigned int Layer::out_features() { return out; }
Node *Layer::node(unsigned int index) { return &n[index]; }

void Net::add_layer(unsigned int in, unsigned int out) { layers.push_back(Layer(in, out)); }
void Net::init(std::default_random_engine &sd) {
    seed = &sd;
    std::normal_distribution<double> gaussian(0.00, 1.00);
    for(unsigned int l = 0; l < layers.size(); l++) {
        for(unsigned int n = 0; n < layers[l].out_features(); n++) {
            for(unsigned int i = 0; i < layers[l].in_features(); i++)
                layers[l].node(n)->set_weight(i, gaussian(*seed) * sqrt(2.00 / layers[l].in_features()));
        }
    }
}

void Net::use_softmax() { softmax = true; }
bool Net::is_softmax() { return softmax; }

unsigned int Net::num_of_layers() { return layers.size(); }
Layer *Net::layer(unsigned int index) { return &layers[index]; }
Layer *Net::back() { return &layers.back(); }

std::vector<double> Net::forward(std::vector<double> &x, bool noise) {
    std::vector<double> yhat; double expsum = 0.00;
    std::normal_distribution<double> gaussian(0.00, 0.05);
    for(unsigned int l = 0; l < layers.size(); l++) {
        for(unsigned int n = 0; n < layers[l].out_features(); n++) {
            double dot = 0.00;
            for(unsigned int i = 0; i < layers[l].in_features(); i++) {
                double weight = layers[l].node(n)->weight(i) + (noise ? gaussian(*seed) : 0.00);
                dot += (l == 0 ? x[i] : layers[l-1].node(i)->act()) * weight;
            }
            dot += layers[l].node(n)->bias() + (noise ? gaussian(*seed) : 0.00);

            layers[l].node(n)->init();
            layers[l].node(n)->set_sum(dot);

            if(l == layers.size() - 1) {
                if(softmax) expsum += exp(layers[l].node(n)->sum());
                else yhat.push_back(layers[l].node(n)->sum());
                continue;
            }

            layers[l].node(n)->set_act(relu(layers[l].node(n)->sum()));
        }
    }

    if(softmax) {
        for(unsigned int n = 0; n < layers.back().out_features(); n++) {
            double norm = exp(layers.back().node(n)->sum()) / expsum;
            yhat.push_back(norm);
        }
    }

    return yhat;
}

void Net::save(std::string &path) {
    std::ofstream out(path);
    if(out.is_open()) {
        for(unsigned int l = 0; l < layers.size(); l++) {
            for(unsigned int n = 0; n < layers[l].out_features(); n++) {
                for(unsigned int i = 0; i < layers[l].in_features(); i++)
                    out << layers[l].node(n)->weight(i) << " ";
                out << layers[l].node(n)->bias() << "\n";
            }
        }
        out.close();
    }
}

void Net::load(std::string &path) {
    std::ifstream out(path);
    if(out.is_open()) {
        for(unsigned int l = 0; l < layers.size(); l++) {
            for(unsigned int n = 0; n < layers[l].out_features(); n++) {
                std::string line;
                std::getline(out, line);
                for(unsigned int i = 0; i < layers[l].in_features(); i++) {
                    double weight = std::stod(line.substr(0, line.find(" ")));
                    layers[l].node(n)->set_weight(i, weight);
                    line = line.substr(line.find(" ") + 1);
                }
                double bias = std::stod(line);
                layers[l].node(n)->set_bias(bias);
            }
        }
        out.close();
    }
}

void copy(Net &src, Net &dst, double tau) {
    bool empty = !dst.num_of_layers();
    for(unsigned int l = 0; l < src.num_of_layers(); l++) {
        unsigned int in = src.layer(l)->in_features();
        unsigned int out = src.layer(l)->out_features();
        if(empty) dst.add_layer(in, out);

        for(unsigned int n = 0; n < out; n++) {
            for(unsigned int i = 0; i < in; i++) {
                double src_weight = src.layer(l)->node(n)->weight(i);
                double dst_weight = dst.layer(l)->node(n)->weight(i);
                dst.layer(l)->node(n)->set_weight(i, tau * src_weight + (1.00 - tau) * dst_weight);
            }
            double src_bias = src.layer(l)->node(n)->bias();
            double dst_bias = dst.layer(l)->node(n)->bias();
            dst.layer(l)->node(n)->set_bias(tau * src_bias + (1.00 - tau) * dst_bias);
        }
    }
    
    if(src.is_softmax()) dst.use_softmax();
}