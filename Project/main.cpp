//
// Created by Anshuman Mishra on 11/25/19.
//

// These statements are saying to include source code that is stored somewhere else (these come with most compilers)
#include <iostream>		// For console i/o
#include <vector>		// For arrays that we don't use for linear algebra
#include <string>		// For strings used in file names
#include <fstream>		// For file i/o
#include <iomanip>		// For setprecision when printing numbers to the console or files


#include <boost/math/special_functions/beta.hpp>
#include <algorithm>
#include <Eigen/Dense>

#include "headers/MathUtils.h"
#include "headers/Policy.h"
#include "headers/Cartpole.h"
#include "headers/FourierBasis.h"
#include "headers/HelperFunctions.hh"

using namespace std;			// Some terms below are "inside" std. For example, cout, cin, and endl. Normally you have to write std::cout. This line makes it so that you don't have to write std:: before everything you are using from standard libraries.
using namespace Eigen;
using namespace boost::math;

void checkpoint(){
  cout << "Checkpoint"; getchar();
}

struct HistoryElement{
  vector<double> state;
  int action;
  double reward;

  HistoryElement (vector<double> s, int a, double r){
    state = s;
    action = a;
    reward = r;
  }
};

double PDIS(const vector<HistoryElement>& history, const Policy piE, const Policy piB, double gamma, const FourierBasis& fb){
//  cout << "Called PDIS for history" << endl;
  double value = 0;
  double gammaCoeff = 1.0;
  double piCoeff = 1.0;
  for (HistoryElement el : history){
//    cout << el.state[0] << el.action << el.reward << endl;
    vector<double> phi_vec = fb.basify(el.state);
    Map<VectorXd> phi = VectorXd::Map(&phi_vec[0], phi_vec.size());
    piCoeff *= (piE.getActionProbability(phi, el.action) / piB.getActionProbability(phi, el.action));
    value += gammaCoeff * piCoeff * el.reward;
//    cout << "Gamma: " << gammaCoeff << " Pi: " << piCoeff << " Value: " << value << endl;
//    getchar();
    gammaCoeff *= gamma;
    if (isnan(value)){
      cout << "PDIS is nan" << endl;
      cout << "piCoeff: " << piCoeff << " gammaCoeff: " << gammaCoeff << " Reward: " << el.reward << endl;
      getchar();
    }
  }
//  cout << "PDIS = " << value << endl;
  return value;
}

VectorXd PDIS(const vector<vector<HistoryElement> >& data, const Policy& piE, const Policy& piB, double gamma, const FourierBasis& fb){
//  cout << "Called PDIS for data" << endl;
  VectorXd pdis(data.size());
  for (int h=0; h<data.size(); h++)
    pdis[h] = PDIS(data[h], piE, piB, gamma, fb);
  return pdis;
}

void splitData(
    const vector<vector<HistoryElement> >& data,
    vector<vector<HistoryElement> >& Dc,
    vector<vector<HistoryElement> >& Ds,
    double ratio){
  int d = data.size();
  int i=0;
  while (i<d*ratio){
    Dc.push_back(data[i++]);
  }
  while (i<data.size()){
    Ds.push_back(data[i++]);
  }
  cout << "Split: " << data.size() << " => " << Dc.size() << "," << Ds.size() << endl;
}

bool candidatePassesTest(const vector<vector<HistoryElement>>& Ds, const Policy piE, const Policy piB, const FourierBasis& fb, const double delta, const double confBound){
  cout << "Testing Candidate" << endl;
  VectorXd j = PDIS(Ds, piE, piB, 1.0, fb); // Get the primary objective
  double ub = ttestLowerBound(j, delta);
  cout << "Target: " << confBound << " UB: " << ub << endl;
  return ub > confBound;
}

// The objective function maximized by getCandidateSolution.
double candidateObjective(
    const VectorXd& theta,                // The solution to evaluate
    const void * params[],                // Other terms that we need to compute the objective value, packed into one object - an array of const pointers to objects of unknown types (unknown to CMA-ES, but known to us here as we packed this object)
    mt19937_64& generator)                // The random number generator to use)
{
//  cout << "Evaluating Candidate" << endl;
  const vector<vector<HistoryElement>>* data = (vector<vector<HistoryElement>>*) params[0];
  Policy* piE = (Policy*) params[1];
  const Policy* piB = (Policy*) params[2];
  const FourierBasis* fb = (FourierBasis*) params[3];
  const int* safetyDataSize = (int*) params[4];
  const double* c = (double*) params[5];
  const double* delta = (double*) params[6];

  piE->setTheta(theta);
//  checkpoint();
  VectorXd j = PDIS(*data, *piE, *piB, 1.0, *fb); // Get the primary objective
  double ub = ttestLowerBound(j, *delta, *safetyDataSize);
  double result;
  cout << " Mean: " << j.mean() << " StdDev: " << stddev(j) << " ttest: " << ub << endl;
  if (ub < *c) {
    cout << "Failed Barrier Function" << endl;
    result = -100;
  }
  else
    result = j.mean();
//  cout << "Result : " << result << endl;
  return result;
}

// Use the provided data to get a solution expected to pass the safety test
VectorXd getCandidateSolution(
    const vector<vector<HistoryElement> >& data,
    const double delta,
    const int safetyDataSize,
    Policy& piE,
    const Policy& piB,
    const FourierBasis& fb,
    const double c,
    mt19937_64 & generator) {
  VectorXd initialSolution = piE.getParams();    // Where should the search start? Let's just use the linear fit that we would get from ordinary least squares linear regression
  double initialSigma = 2.0*(initialSolution.dot(initialSolution) + 1.0); // A heuristic to select the width of the search based on the weight magnitudes we expect to see.
  int numIterations = 100;                          // Number of iterations that CMA-ES should run. Larger is better, but takes longer.
  bool minimize = false;                            // We want to maximize the candidate objective.
  // Pack parameters of candidate objective into params. In candidateObjective we need to unpack in the same order.
  const void* params[7];

  params[0] = &data;
  params[1] = &piE;
  params[2] = &piB;
  params[3] = &fb;
  params[4] = &safetyDataSize;
  params[5] = &c;
  params[6] = &delta;

  cout << "Calling CMAES" << endl;
  // Use CMA-ES to get a solution that approximately maximizes candidateObjective
  return CMAES(initialSolution, initialSigma, numIterations, candidateObjective, params, minimize, generator);
}

double getTarget(const vector<vector<HistoryElement>>& data){
  double gamma = 1.0;
  int numEpisodes = data.size();
  VectorXd returns(data.size());
  for (int e = 0; e<data.size(); e++){
    double episodeReturn = 0.0;
    double gammaCoeff = 1.0;
    for (HistoryElement he : data[e]){
      episodeReturn += gammaCoeff*he.reward;
      gammaCoeff *= gamma;
    }
    returns[e] = episodeReturn;
  }
  return 1.1*returns.mean();
}

void generateData(int numEpisodes, int maxEpisodeLength){
  cout << "Generating Data" << endl;
  // Generate Histories for some policy
  // Write to file
  ofstream out("../../../output/data.csv");
  Cartpole e;
  int stateDim = e.getStateDim();
  int order = 3;
  int numActions = e.getNumActions();
  FourierBasis fb;
  fb.init(stateDim, 0, order);

  out << stateDim << endl;
  out << numActions << endl;
  out << order << endl;

  Policy p(numActions, fb.getNumOutputs(), 1234);
  VectorXd params = p.getParams();
  for (int i=0; i<params.size(); i++){
    out << params[i] << ",";
  }
  out << endl;

  out << numEpisodes << endl;

  mt19937_64 gen(0);
  double gamma = 1.0;

  for (int eps = 0; eps < numEpisodes; eps++){
    double curGamma = 1.0;					// We plot the discounted return - this stores gamma^t, which starts at 1.
    bool inTerminalState = false;			// We will use this flag to determine when we should terminate the loop below. If environment[trial].inTerminalState() is slow to call, this saves us from calling it a couple times. For our MDPs it really doesn't matter that we're doing this more efficiently.
    e.newEpisode(gen);	// Reset the environment, telling it to start a new episode.
    vector<double> state = e.getState(gen);	// Get the initial state.
    for (int t = 0; (t < maxEpisodeLength) && (!inTerminalState); t++) { // Loop over time steps in the episode, stopping when we hit the max episode length or when we enter a terminal state.
      vector<double> phi_vec = fb.basify(state);
      VectorXd phi = VectorXd::Map(&phi_vec[0], phi_vec.size());
      int action = p.getAction(phi);
      double reward = e.update(action, gen); // Apply the action by updating the environment with the chosen action, and get the resulting reward.

      for (double val : state)
        out << val << ",";
      out << action << ",";
      out << reward;

      vector<double> nextState = e.getState(gen); // Get the resulting state of the environment from this transition
      inTerminalState = e.inTerminalState(); // Store whether this is next-state is a terminal state.

      if ((t < maxEpisodeLength-1) && (!inTerminalState))
        out << ",";

      state = nextState; // Prepare for the next iteration of the loop with this line and the next.
      curGamma *= gamma;
    }
    out<<endl;
  }

  cout << "Data Generation Complete" << endl;
}

// Given a filename, load the file into an MDP, run sanity checks on this MDP, run value iteration, and print the result to a file.
void run() {

//  generateData(100000, 10);

  /*********************** End of Data Prep ***********************/

  cout << "Reading Data" << endl;
//  ifstream in("../../../output/data.csv");
  ifstream in("../../../input/data.csv");

  int stateDim, numActions, order;
  in >> stateDim;
//  cout << "stateDim: " << stateDim << endl;
  in >> numActions;
//  cout << "numActions: " << numActions << endl;
  in >> order;
//  cout << "order: " << order << endl;

  FourierBasis fb;
  fb.init(stateDim, 0, order);

  VectorXd thetaB(numActions*fb.getNumOutputs());
//  cout << "ThetaB " << thetaB.size() << endl;

  cout << "Reading Policy" << endl;
  string s;
  for (int i=0; i<thetaB.size(); i++){
      getline(in, s, ',');
      thetaB[i] = stod(s);
//      cout << thetaB[i] << ",";
  }
  cout << endl;
  // Complete reading the line
//  getline(in, s);

  Policy piB(thetaB, numActions, fb.getNumOutputs(), 123);

  int numEpisodes;
  in >> numEpisodes;
//  cout << "numEpisodes: " << numEpisodes << endl;

  vector<vector<HistoryElement> > data;
  for (int eps = 0; eps < numEpisodes; eps++){
    vector<HistoryElement> episode;
//    cout << "Reading Episode " << eps << endl;
    string ep;
    in >> ep;
//    cout << ep << endl;

    stringstream ss(ep);
    while(ss.good()){
      vector<double> state(stateDim, 0.0);
//      cout << "State: ";
      for (int st = 0; st < stateDim; st++) {
        getline(ss, s, ',');
        state[st] = stod(s);
//        cout << state[st] << ",";
      }
//      cout << " Action: ";

      getline(ss, s, ',');
      int action = stoi(s);
//      cout << action << ", Reward: ";
//      cout << action << ",";

      getline(ss, s, ',');
      double reward = stod(s);
//      cout << reward << ",";
//      cout << endl;
//      cin >> s;
      episode.push_back(HistoryElement(state, action, reward));
    }
//    cout << endl;
    data.push_back(episode);
  }

  cout << "Data Read Complete" << endl;

  /*********************** End of Data Read ***********************/

  // Split Data into Ds and Dc
  vector<vector<HistoryElement> > Ds, Dc;
  splitData(data, Dc, Ds, 0.6);

  int numPolicies = 10;
  vector<Policy> policies;

  mt19937_64 generator(123);
  double target = getTarget(data);
  double delta = 0.1;
  Policy piE(numActions, fb.getNumOutputs(), 123);

  // Loop while more policies need to be found
  while(policies.size() < numPolicies){
    //   Select Candidate Policy
    VectorXd candidate = getCandidateSolution(
          Dc,
          delta,
          Ds.size(),
          piE,
          piB,
          fb,
          target,
          generator);

    cout << "Generated Candidate" << endl;
    piE.setTheta(candidate);
    //   Store Candidate Policy if Test Passed
    if (candidatePassesTest(Ds, piE, piB, fb, delta, target)){
      cout << "****************** CANDIDATE PASSED TEST ******************" << endl;
      policies.push_back(piE);
    } else
      cout << "****************** CANDIDATE FAILED TEST ******************" << endl;
    checkpoint();
  }

  // Write the policies to result
  ofstream policyOut("../../../output/result.csv");
  for (auto policy : policies){
    VectorXd params = policy.getParams();
    for (int i=0; i<params.size(); i++){
      policyOut << params[i] << ",";
    }
    policyOut << endl;
  }
}

int main(int argc, char* argv[])
{
  run();
//  VectorXd a(4);
//  a  << 1.0, 2.0, 3.0, 4.0;
//  cout << a << endl;
//  MatrixXd m = MatrixXd::Map(a.data(), 2,2);
//  cout << m << endl;
//
//  VectorXd b(4);
//  b << 2.0,3.0,4.0,5.0;
//  m = MatrixXd::Map(b.data(), m.rows(), m.cols());
//  cout << m << endl;
  cout << "Done." << endl;
}