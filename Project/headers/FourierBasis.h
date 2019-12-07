//
// Created by Anshuman Mishra on 11/26/19.
//

#ifndef CLION_FOURIERBASIS_H
#define CLION_FOURIERBASIS_H

#include <vector>
using namespace std;

// A class implementing the Fourier basis
class FourierBasis
{
public:
  void init(const int & inputDimension, int iOrder, int dOrder);
  int getNumOutputs() const;
  vector<double> basify(const vector<double> & x) const;

private:
  int nTerms;							// Total number of outputs
  int inputDimension;
  vector<vector<double> > c;	// Coefficients
};

#endif // CLION_FOURIERBASIS_H
