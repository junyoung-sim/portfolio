#ifndef __QUANT_HPP_
#define __QUANT_HPP_

#include <cstdlib>
#include <vector>

#include "../lib/net.hpp"

class Memory
{
private:
    std::vector<double> s0;
    std::vector<double> a;
    std::vector<double> s1;
    double r;
public:
    Memory() {}
    Memory(std::vector<double> &current, std::vector<double> &action, std::vector<double> &next, double reward) {
        s0.swap(current);
        a.swap(action);
        s1.swap(next);
        r = reward;
    }
    ~Memory() {
        std::vector<double>().swap(s0);
        std::vector<double>().swap(a);
        std::vector<double>().swap(s1);
    }

    std::vector<double> *state();
    std::vector<double> *action();
    std::vector<double> *next_state();
    double reward();
};

class DDPG
{
private:
    Net *actor, target_actor;
    Net *critic, target_critic;
public:
    DDPG() {}
    DDPG(Net &a, Net &c) {
        actor = &a;
        critic = &c;
        copy(*actor, target_actor, 1.00);
        copy(*critic, target_critic, 1.00);
    }
    ~DDPG() {}

    std::vector<double> epsilon_greedy(std::vector<double> &state, double eps);

    void optimize_critic(std::vector<double> &state_action, double q, double optimal, std::vector<double> &agrad, double alpha, double lambda);
    void optimize_actor(std::vector<double> &state, std::vector<double> &action, std::vector<double> &agrad, double alpha, double lambda);
    double optimize(Memory &memory, double gamma, double alpha, double lambda);

    void sync(double tau);
};

#endif