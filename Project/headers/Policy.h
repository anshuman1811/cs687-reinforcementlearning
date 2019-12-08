//
// Created by Anshuman Mishra on 11/26/19.
//

#ifndef CLION_POLICY_H
#define CLION_POLICY_H

#include <boost/math/special_functions/beta.hpp>
#include <random>
#include <algorithm>
#include <Eigen/Dense>

using namespace std;			// Some terms below are "inside" std. For example, cout, cin, and endl. Normally you have to write std::cout. This line makes it so that you don't have to write std:: before everything you are using from standard libraries.
using namespace Eigen;

class Policy{
public:
  Policy (int numActions_, int stateTerms_, unsigned seed);
  Policy (VectorXd& theta_init, int numActions_, int stateTerms_, unsigned seed);
  Policy(Policy pi, unsigned seed);
  int getAction (const VectorXd& phi);
  double getActionProbability (const VectorXd& phi, int action) const;
  VectorXd getTheta() const;
  void setTheta (const VectorXd& newTheta);
  int getNumActions() const;
  int getStateTerms() const;

private:
  MatrixXd theta;
  default_random_engine gen;
  int numActions;
  int stateTerms;
};

#endif // CLION_POLICY_H
