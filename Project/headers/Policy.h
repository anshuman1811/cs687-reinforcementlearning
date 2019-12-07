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
  Policy (int numActions, int stateTerms, unsigned seed);
  Policy (VectorXd& theta_init, int numActions, int stateTerms, unsigned seed);
  int getAction (const VectorXd& phi);
  double getActionProbability (const VectorXd& phi, int action) const;
  VectorXd getParams();
  void setTheta (const VectorXd& newTheta);

private:
  MatrixXd theta;
  default_random_engine gen;
};

#endif // CLION_POLICY_H
