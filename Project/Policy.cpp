//
// Created by Anshuman Mishra on 11/26/19.
//

#include "headers/Policy.h"
#include <algorithm>
#include <random>
#include <functional>
#include <boost/math/special_functions/beta.hpp>
#include "headers/MathUtils.h"

using namespace std;
using namespace Eigen;

Policy::Policy (int numActions_, int stateTerms_, unsigned seed){
//  cout << "Initializing without theta" << endl;
//  cout << "Theta before " << theta.size() << endl;
  // init theta
  numActions = numActions_;
  stateTerms = stateTerms_;
  theta = MatrixXd(numActions, stateTerms);
  //  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  gen = default_random_engine (seed);
//  cout << "Theta after " << theta.size() << endl;
}

Policy::Policy (VectorXd& theta_init, int numActions_, int stateTerms_, unsigned seed){
  numActions = numActions_;
  stateTerms = stateTerms_;
//  cout << "Initializing with theta " << theta_init.size() << endl;
//  cout << "Theta before " << theta.size() << endl;
  theta = MatrixXd::Map(theta_init.data(), numActions, stateTerms);
//  cout << "Theta after " << theta.size() << endl;
  gen = default_random_engine (seed);
}

Policy::Policy(Policy pi, unsigned seed){
  numActions = pi.getNumActions();
  stateTerms = pi.getStateTerms();
  theta = MatrixXd::Map(pi.getTheta().data(), numActions, stateTerms);
  gen = default_random_engine (seed);
}

void Policy::setTheta(const VectorXd& newTheta) {
//  cout << "Setting theta " << newTheta.size() << endl;
//  cout << "Theta before " << theta.size() << endl;
  theta = MatrixXd::Map(newTheta.data(), theta.rows(), theta.cols());
//  cout << "Theta after " << theta.size() << endl;
}

VectorXd Policy::getTheta() const{
  return VectorXd::Map(theta.data(), theta.size());
}

int Policy::getNumActions() const{
  return numActions;
}

int Policy::getStateTerms() const{
  return stateTerms;
}

// Softmax Action Selection with Linear Function Approximation
int Policy::getAction (const VectorXd& phi){
//  cout << "Getting Action" << endl;
//  cout << "Phi " << phi << endl;
  VectorXd dot = phi.transpose()*theta;
  VectorXd q = exp(dot.array() - dot.maxCoeff());
//  cout << "q " << q << endl;

  discrete_distribution<int> dist(q.data(), q.data() + q.rows() * q.cols());
  int action = dist(gen);
//  cout << "Action:" << action << endl;
  return action;
}

double Policy::getActionProbability (const VectorXd& phi, int action) const {
//  cout << "Getting Action Probability";
//  cout << theta.rows() << "x" << theta.cols() << "dot" << phi.rows() << "x" << phi.cols() << endl;
//  cout << "Phi " << phi.transpose() << endl;
//  cout << "theta " << theta << endl;
  VectorXd dot = phi.transpose()*theta;
//  cout << "dot " << dot.transpose() << endl;
  VectorXd q = exp(dot.array() - dot.maxCoeff());

  discrete_distribution<int> dist(q.data(), q.data() + q.rows() * q.cols());
  double p = dist.probabilities()[action];
  if (isnan(p))
    p = 1.0;
//  if (isnan(p)){
//    cout << "Action Probability is nan" << endl;
//    cout << "phi: " << phi.transpose() << endl;
//    cout << "Q: " << q.transpose() << endl;
//    cout << "action: " << action << endl;
//    getchar();
//  }
//  cout << "p=" << p << endl;
  return p;
}