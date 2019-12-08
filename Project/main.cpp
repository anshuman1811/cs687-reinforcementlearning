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
#include "headers/Gridworld.hpp"
#include "headers/Walk.h"
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
  VectorXd initialSolution = piE.getTheta();
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
//    if (data[e].size() > 10 || episodeReturn < -10)
//      checkpoint();
  }
  double avgReturn = returns.mean();
  return (avgReturn > 0 ? 1.1 : 0.9)*avgReturn;
}

template <typename Environment>
double getAverageReturn(Environment& e, Policy& pi, int numEpisodes, int maxEpisodeLength, const FourierBasis& fb){
//  Cartpole e;
  mt19937_64 gen(0);
  double gamma = 1.0;
  VectorXd returns(numEpisodes);

  for (int eps = 0; eps < numEpisodes; eps++){
    double curGamma = 1.0;					// We plot the discounted return - this stores gamma^t, which starts at 1.
    bool inTerminalState = false;			// We will use this flag to determine when we should terminate the loop below. If environment[trial].inTerminalState() is slow to call, this saves us from calling it a couple times. For our MDPs it really doesn't matter that we're doing this more efficiently.
    e.newEpisode(gen);	// Reset the environment, telling it to start a new episode.
    vector<double> state = e.getState(gen);	// Get the initial state.
    double episodeReturn = 0.0;
    for (int t = 0; (t < maxEpisodeLength) && (!inTerminalState); t++) { // Loop over time steps in the episode, stopping when we hit the max episode length or when we enter a terminal state.
      vector<double> phi_vec = fb.basify(state);
      VectorXd phi = VectorXd::Map(&phi_vec[0], phi_vec.size());
      int action = pi.getAction(phi);
      double reward = e.update(action, gen); // Apply the action by updating the environment with the chosen action, and get the resulting reward.

      vector<double> nextState = e.getState(gen); // Get the resulting state of the environment from this transition
      inTerminalState = e.inTerminalState(); // Store whether this is next-state is a terminal state.

      episodeReturn += curGamma*reward;
      state = nextState; // Prepare for the next iteration of the loop with this line and the next.
      curGamma *= gamma;
    }
    returns[eps] = episodeReturn;
  }
  cout << "AvgReturn: " << returns.mean() << endl;
  return returns.mean();
}

template <typename Environment>
void generateData(Environment& e, int numEpisodes, int maxEpisodeLength, int order){
  cout << "Generating Data" << endl;
  // Generate Histories for some policy
  // Write to file
  ofstream out("../../../output/data.csv");
//  Cartpole e;
  int stateDim = e.getStateDim();
//  int order = 3;
  int numActions = e.getNumActions();
  FourierBasis fb;
  fb.init(stateDim, 0, order);
  mt19937_64 gen(0);

  out << stateDim << endl;
//  cout << "StateDim in env " << stateDim << " " << e.getState(gen).size() << endl;
//  checkpoint();
  out << numActions << endl;
  out << order << endl;

  Policy p(numActions, fb.getNumOutputs(), 1234);
//  VectorXd params(numActions*fb.getNumOutputs());
//  params << 0.452687,-0.123691,-0.187805,-0.0419247,-0.3103,0.0169222,-0.440849,0.171773,0.122356,-0.0208773,-0.377911,0.24458,-0.345339,0.221081,0.181478,0.0931315,0.316947,-0.237248,-0.0783118,0.102174,0.197309,0.0129477,0.237053,-0.283958,0.0931003,-0.159108,-0.11468,-0.0336945,-0.312665,0.0494717,-0.437631,0.185128,0.13921,0.124541,0.269549,-0.0445983,-0.311025,0.186034,0.181678,0.065481,0.311506,-0.245937,0.320217,-0.266973,-0.10468,-0.0398326,0.24824,-0.237777,0.0789096,-0.109312,-0.131987,-0.0215236,-0.190011,0.263565,-0.391198,0.144332,0.109239,0.0786274,0.245533,-0.0450382,0.407343,-0.221437,-0.0711912,-0.023617,0.332143,-0.184834,0.315291,-0.193174,-0.122204,-0.08163,-0.262694,0.218761,0.136993,-0.0538062,-0.220642,0.0300469,-0.256064,0.217929,-0.118554,0.118296,0.102564,0.289925,-0.14454,0.0308677,-0.036806,-0.119257,-0.000854144,-0.27288,0.174216,-0.101032,0.0331704,-0.236314,0.194932,-0.219322,0.183163,-0.00102567,0.0264519,0.160525,-0.151711,-0.0274355,0.101995,0.0911902,-0.0420769,0.179202,-0.249171,0.0371926,-0.122774,-0.0147314,-0.0374526,-0.1332,0.0527589,-0.267015,0.200302,-0.0525608,0.117418,0.0845154,-0.0264594,-0.182052,0.195832,0.0153543,0.0274128,0.176015,-0.22169,0.192534,-0.244502,0.0506698,-0.0728419,0.158957,-0.293461,-0.0106893,-0.141654,-0.00546437,0.0100509,-0.113205,0.273374,-0.185785,0.167286,-0.0742353,0.0805248,0.0537695,-0.0490843,0.207685,-0.225989,0.101884,-0.0554201,0.15327,-0.205815,0.138937,-0.206245,0.060542,-0.052369,-0.0965512,0.20772,-0.0393728,-0.0666479,-0.00757391,0.10165,-0.104987,0.26088,0.0429262,0.137972,-0.0754444;
//  p.setTheta(params);
  VectorXd params = p.getTheta();
  for (int i=0; i<params.size(); i++){
    out << params[i] << ",";
  }
  out << endl;

  out << numEpisodes << endl;

  double gamma = 1.0;

  vector<HistoryElement> firstEps;
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

      if (eps == 0)
        firstEps.push_back(HistoryElement(state, action, reward));

      vector<double> nextState = e.getState(gen); // Get the resulting state of the environment from this transition
      inTerminalState = e.inTerminalState(); // Store whether this is next-state is a terminal state.

      if ((t < maxEpisodeLength-1) && (!inTerminalState))
        out << ",";

      state = nextState; // Prepare for the next iteration of the loop with this line and the next.
      curGamma *= gamma;
    }
    out<<endl;
  }

  for (int t=0; t<firstEps.size(); t++){
    HistoryElement ht = firstEps[t];
    vector<double> phi_vec = fb.basify(ht.state);
    VectorXd phi = VectorXd::Map(&phi_vec[0], phi_vec.size());
    out<< p.getActionProbability(phi, ht.action);
    out << ((t==firstEps.size()-1) ? '\n' : ',');
  }

  cout << "Data Generation Complete" << endl;
}

vector<Policy> HCOPE(
    vector<vector<HistoryElement>>& data,
    const int numPolicies,
    const double delta,
    const double target,
    const Policy& piB,
    const FourierBasis& fb,
    const string outputDir){
  // Split Data into Ds and Dc
  vector<vector<HistoryElement> > Ds, Dc;
  double dataSplitRatio = 0.6;
  splitData(data, Dc, Ds, dataSplitRatio);

//  int numPolicies = 100;

  mt19937_64 generator(123);
  cout << "Target: " << target << endl;
//  Policy piE(numActions, fb.getNumOutputs(), 123);
//  Policy piE (piB, 123);

//  VectorXd theta(numActions*fb.getNumOutputs());
//  theta << 1, 1, 0.01, -0.01;
//  piE.setTheta(theta);
//  VectorXd pdis = PDIS(data, piE, piB, 1.0, fb);
//  cout << "PDIS " << pdis.mean() << endl;
//
//  pdis = PDIS(data, piB, piB, 1.0, fb);
//  cout << "PDIS " << pdis.mean() << endl;
//
//  cout << "Mean Return in Data " << target/1.1 << endl;
//  checkpoint();

  // Loop while more policies need to be found
//  ofstream policyOut("../../../output/result.csv");

  vector<Policy> policies(numPolicies, Policy(piB, 123));
  #pragma omp parallel for
  for (int p=0; p<numPolicies; p++){
//  while(policies.size() < numPolicies){
    bool passedTest = false;
    VectorXd candidate;
    while(!passedTest) {
      //   Select Candidate Policy
      policies[p].setTheta(piB.getTheta());
      candidate = getCandidateSolution(Dc, delta, Ds.size(), policies[p], piB,
                                       fb, target, generator);

      cout << "Generated Candidate" << endl;
      policies[p].setTheta(candidate);

      passedTest = candidatePassesTest(Ds, policies[p], piB, fb, delta, target);
      if (passedTest)
        cout << "****************** CANDIDATE PASSED TEST ******************" << endl;
      else
        cout << "****************** CANDIDATE FAILED TEST ******************" << endl;
    }
    //   Store Candidate Policy if Test Passed
    ofstream policyOut(outputDir + to_string(p) + ".csv");
    for (int i=0; i<candidate.size(); i++){
      policyOut << candidate[i];
      policyOut << (i == candidate.size()-1 ? "\n":",") << flush;
      cout << candidate[i];
      cout << (i == candidate.size()-1 ? "\n":",");
    }
    //checkpoint()
  }
  return policies;
}

void readData(
    string filename,
    int& stateDim,
    int& numActions,
    int& order,
    Policy& piB,
    int& numEpisodes,
    FourierBasis& fb,
    vector<vector<HistoryElement> >& data){
  cout << "Reading Data" << endl;
//  ifstream in("../../../output/data.csv");
//  ifstream in("../../../input/data.csv");
  ifstream in(filename);

  in >> stateDim;
  in >> numActions;
  in >> order;

  fb.init(stateDim, 0, order);

  VectorXd thetaB(numActions*fb.getNumOutputs());
//  cout << "ThetaB " << thetaB.size() << endl;

  cout << "Reading Policy" << endl;
  string s;
  for (int i=0; i<thetaB.size(); i++){
    char delim = (i == thetaB.size()-1 ? '\n': ',');
    getline(in, s, delim);
    thetaB[i] = stod(s);
//    cout << thetaB[i] << delim;
  }
  // Complete reading the line
//  getline(in, s);

  piB = Policy(thetaB, numActions, fb.getNumOutputs(), 123);

  in >> numEpisodes;

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

  cout << "Validating Policy Representation" << endl;
  string valPi;
  in >> valPi;
  stringstream ss(valPi);
  VectorXd expectedPi(data[0].size());
  VectorXd actualPi(data[0].size());
  bool match = true;
  for (int t=0; t<data[0].size(); t++){
    HistoryElement ht = data[0][t];
    vector<double> phi_vec = fb.basify(ht.state);
    VectorXd phi = VectorXd::Map(&phi_vec[0], phi_vec.size());
    actualPi[t] = piB.getActionProbability(phi, ht.action);

    string s;
    getline(ss, s, ',');
//    cout << s << ", ";
    expectedPi[t] = stod(s);

    if (abs(expectedPi[t] - actualPi[t]) > 1e-5)
      match = false;
  }

  if (!match){
    cout << "Policies Dont Match!"  << endl;
    cout << "Expected: " << expectedPi.transpose() << endl;
    cout << "Actual: " << actualPi.transpose() << endl;
    checkpoint();
    return;
  }

  cout << "Data Read Complete" << endl;
}

void run() {

  bool test = true;

  /****************************************************************/

  bool genData = false;
  int genEpisodes = 100000, genMaxEpisodeLength = 15, genOrder = 1;
//  Cartpole genEnv;
  Gridworld genEnv;
//  Walk genEnv;
  if (test && genData)
    generateData(genEnv, genEpisodes, genMaxEpisodeLength, genOrder);

  /*********************** End of Data Prep ***********************/

  int stateDim, numActions, order, numEpisodes;
  Policy piB(0,0,0);
  FourierBasis fb;
  vector<vector<HistoryElement> > data;
  if (test)
    readData("../../../output/data.csv", stateDim, numActions, order, piB, numEpisodes, fb, data);
  else
    readData("../../../input/data.csv", stateDim, numActions, order, piB, numEpisodes, fb, data);

  cout << "stateDim: " << stateDim << endl;
  cout << "numActions: " << numActions << endl;
  cout << "order: " << order << endl;
  cout << "Policy: " << piB.getTheta().transpose() << endl;
  cout << "numEpisodes: " << numEpisodes << endl;

  /*********************** End of Data Read ***********************/

//  VectorXd testTheta(piB.getTheta().size());
//  testTheta << 0.33304397, -1.3536084, -0.391217, -2.93140976, -4.07265164, -2.13137546, -1.94398795, 2.53769193;
//  Policy testPolicy (piB);
//  testPolicy.setTheta(testTheta);
//  double avgReturn = getAverageReturn(genEnv, testPolicy, genEpisodes, genMaxEpisodeLength, fb);
//  return;

  /********************* End of Testing Space *********************/

  int numPolicies = 50;
  double delta = 0.05;
  double target = getTarget(data);
  string outputDir;
  if (test)
    outputDir = "../../../result/" + to_string(delta) + "_";
  else
    outputDir = "../../../final_result/";
  vector<Policy> policies = HCOPE(data, numPolicies, delta, target, piB, fb, outputDir);

  /********************* End of Policy Search *********************/

  if (test) {
      // Validate policies found
      cout << "Validating Result Policies" << endl;
      int worked = 0;
      for (auto policy : policies){
        double avgReturn = getAverageReturn(genEnv, policy, genEpisodes, genMaxEpisodeLength, fb);
        if (avgReturn >= target/1.1)
          worked++;
      }

      cout << "Reality Check! #Policies that worked: " << worked << "/" << policies.size() << endl;
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