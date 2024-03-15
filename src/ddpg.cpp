#include <cstdlib>
#include <vector>

#include "../lib/ddpg.hpp"

std::vector<double> *Memory::state() { return &s0; }
std::vector<double> *Memory::action() { return &a; }
std::vector<double> *Memory::next_state() { return &s1; }
double Memory::reward() { return r; }

std::vector<double> DDPG::epsilon_greedy(std::vector<double> &state, double eps) {
    double explore = (double)rand() / RAND_MAX;
    return actor->forward(state, explore < eps);
}

void DDPG::optimize_critic(std::vector<double> &state_action, double q, double optimal, std::vector<double> &agrad, double alpha, double lambda) {
    for(int l = critic->num_of_layers() - 1; l >= 0; l--) {
        double part = 0.00, grad = 0.00;
        for(unsigned int n = 0; n < critic->layer(l)->out_features(); n++) {
            if(l == critic->num_of_layers() - 1) part = -2.00 * (optimal - q);
            else part = critic->layer(l)->node(n)->err() * drelu(critic->layer(l)->node(n)->sum());

            double updated_bias = critic->layer(l)->node(n)->bias() - alpha * part;
            critic->layer(l)->node(n)->set_bias(updated_bias);

            for(unsigned int i = 0; i < critic->layer(l)->in_features(); i++) {
                if(l == 0) {
                    grad = part * state_action[i];
                    if(i < agrad.size())
                        agrad[i] = part * critic->layer(l)->node(n)->weight(i);
                }
                else {
                    grad = part * critic->layer(l-1)->node(i)->act();
                    critic->layer(l-1)->node(i)->add_err(part * critic->layer(l)->node(n)->weight(i));
                }

                grad += lambda * critic->layer(l)->node(n)->weight(i);

                double updated_weight = critic->layer(l)->node(n)->weight(i) - alpha * grad;
                critic->layer(l)->node(n)->set_weight(i, updated_weight);
            }
        }
    }
}

void DDPG::optimize_actor(std::vector<double> &state, std::vector<double> &action, std::vector<double> &agrad, double alpha, double lambda) {
    for(int l = actor->num_of_layers() - 1; l >= 0; l--) {
        double part = 0.00, grad = 0.00;
        for(unsigned int n = 0; n < actor->layer(l)->out_features(); n++) {
            if(l == actor->num_of_layers() - 1) part = agrad[n] * action[n] * (1.00 - action[n]);
            else part = actor->layer(l)->node(n)->err() * drelu(actor->layer(l)->node(n)->sum());

            double updated_bias = actor->layer(l)->node(n)->bias() + alpha * part;
            actor->layer(l)->node(n)->set_bias(updated_bias);

            for(unsigned int i = 0; i < actor->layer(l)->in_features(); i++) {
                if(l == 0) grad = part * state[i];
                else {
                    grad = part * actor->layer(l-1)->node(i)->act();
                    actor->layer(l-1)->node(i)->add_err(part * actor->layer(l)->node(n)->weight(i));
                }

                grad += lambda * actor->layer(l)->node(n)->weight(i);

                double updated_weight = actor->layer(l)->node(n)->weight(i) + alpha * grad;
                actor->layer(l)->node(n)->set_weight(i, updated_weight);
            }
        }
    }
}

double DDPG::optimize(Memory &memory, double gamma, double alpha, double lambda) {
    std::vector<double> *state = memory.state();
    std::vector<double> *action = memory.action();

    std::vector<double> state_action;
    state_action.insert(state_action.end(), action->begin(), action->end());
    state_action.insert(state_action.end(), state->begin(), state->end());

    std::vector<double> *next_state = memory.next_state();
    std::vector<double> next_state_action = target_actor.forward(*next_state, false);
    next_state_action.insert(next_state_action.end(), next_state->begin(), next_state->end());

    std::vector<double> q = critic->forward(state_action, false);
    std::vector<double> future = target_critic.forward(next_state_action, false);
    double optimal = memory.reward() + gamma * future[0];

    std::vector<double> agrad(action->size(), 0.00);
    
    optimize_critic(state_action, q[0], optimal, agrad, alpha, lambda);
    optimize_actor(*state, *action, agrad, alpha, lambda);

    return q[0];
}

void DDPG::sync(double tau) {
    copy(*actor, target_actor, tau);
    copy(*critic, target_critic, tau);
}