//
// Created by Anshuman Mishra on 11/26/19.
//
#include "headers/Cartpole.h"
#include "headers/MathUtils.h"
#include <cmath>

using namespace std;

Cartpole::Cartpole() {
  mt19937_64 generator(0);
  newEpisode(generator);
}

int Cartpole::getStateDim() const {
  return 4;
}

int Cartpole::getNumActions() const {
  return 2;
}

double Cartpole::update(const int & action, mt19937_64 & generator) {
  double F = action*uMax + (action - 1)*uMax, omegaDot, vDot, subDt = dt / (double)simSteps;
  for (int i = 0; i < simSteps; i++) {
    omegaDot = (g*sin(theta) + cos(theta)*(muc*sign(v) - F - m*l*omega*omega*sin(theta)) / (m + mc) - mup*omega / (m*l)) / (l*(4.0 / 3.0 - m / (m + mc)*cos(theta)*cos(theta)));
    vDot = (F + m*l*(omega*omega*sin(theta) - omegaDot*cos(theta)) - muc*sign(v)) / (m + mc);
    theta += subDt*omega;
    omega += subDt*omegaDot;
    x += subDt*v;
    v += subDt*vDot;
    theta = wrapPosNegPI(theta);
    t += subDt;
  }
  x = bound(x, xMin, xMax);
  v = bound(v, vMin, vMax);
  theta = bound(theta, thetaMin, thetaMax);
  omega = bound(omega, omegaMin, omegaMax);
  return 1;
}

vector<double> Cartpole::getState(mt19937_64 & generator) {
  vector<double> result(4);
  result[0] = normalize(x, xMin, xMax);
  result[1] = normalize(v, vMin, vMax);
  result[2] = normalize(theta, thetaMin, thetaMax);
  result[3] = normalize(omega, omegaMin, omegaMax);
  return result;
}

bool Cartpole::inTerminalState() const {
  return ((fabs(theta) > M_PI / 15.0) || (fabs(x) >= 2.4) || (t >= 20.0 + 10 * dt));
}

void Cartpole::newEpisode(mt19937_64 & generator) {
  theta = omega = v = x = t = 0;
}