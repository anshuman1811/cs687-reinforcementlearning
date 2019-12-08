//
// Created by Anshuman Mishra on 12/7/19.
//
#include "headers/Walk.h"
#include "headers/MathUtils.h"

using namespace std;

/*
This file contains the implementations of all of the functions defined in the Gridworld.hpp file.
*/

// Implement the constructor.
Walk::Walk() {
  mt19937_64 generator(0);	// Initialize the RNG.
  newEpisode(generator);		// Start a new episode.
}

int Walk::getStateDim() const {
  return 1;			// The state will be a tabular representation, implemented using linear function approxiamtion.
}

int Walk::getNumActions() const {
  return 2;					// up/down/left/right
}

double Walk::update(const int & action, mt19937_64 & generator) {
  // Actions correspond to up/down/left/right, where (0,0) is bottom left. Actions always succeed
  if (action == 0)
    x = max(0, x-1);
  else
    x++;
//	return -1;	// Reward is always -1
  return inTerminalState()?1.0:0.0;
}

vector<double> Walk::getState(mt19937_64 & generator) {
  return vector<double>(1, ((double)x)/size);
}

bool Walk::inTerminalState() const {
  return (x == size - 1);	// Are we in state (size-1)?
}

void Walk::newEpisode(mt19937_64 & generator) {
  x = 0;								// Always start in state (0).
}