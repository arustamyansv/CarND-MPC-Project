#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"
#include <limits>

using CppAD::AD;

// TODO: Set the timestep length and duration
size_t N = 10;
double dt = 0.1;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// set speed penalty
double ref_v = 60;

// set the beginning index for each of value types in fg
size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // init cost as 0. Increment it through the process
    fg[0] = 0.0;

    // The part of the cost based on the reference state.
    for (auto t = 0; t < N; t++) {
      fg[0] += CppAD::pow(vars[cte_start + t], 2);
      fg[0] += CppAD::pow(vars[epsi_start + t], 2);
      fg[0] += CppAD::pow(vars[v_start + t] - ref_v, 2);
    }

    // Minimize the use of actuators.
    for (auto t = 0; t < N - 1; t++) {
      fg[0] += 10*CppAD::pow(vars[delta_start + t], 2);
      fg[0] += 10*CppAD::pow(vars[a_start + t], 2);
    }

    // Minimize the value gap between sequential actuations.
    for (auto t = 0; t < N - 2; t++) {
      fg[0] += 100*CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
      fg[0] += 50*CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
    }

    // initialise state
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    AD<double> x0, y0, psi0, v0, cte0, epsi0;
    AD<double> x1, y1, psi1, v1, cte1, epsi1;
    AD<double> delta0, a0, f0, psides0;

    for (int t = 1; t < N; t++) {
      // The state at time t
      x0 = vars[x_start + t - 1];
      y0 = vars[y_start + t - 1];
      psi0 = vars[psi_start + t - 1];
      v0 = vars[v_start + t - 1];
      cte0 = vars[cte_start + t - 1];
      epsi0 = vars[epsi_start + t - 1];

      // Actuations at a time t
      delta0 = vars[delta_start + t - 1];
      a0 = vars[a_start + t - 1];

      f0 = coeffs[0] + coeffs[1] * x0;
      psides0 = CppAD::atan(coeffs[1]);

      // The state at time t+1
      x1 = vars[x_start + t];
      y1 = vars[y_start + t];
      psi1 = vars[psi_start + t];
      v1 = vars[v_start + t];
      cte1 = vars[cte_start + t];
      epsi1 = vars[epsi_start + t];

      // calculate state values for t+2
      fg[1 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[1 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[1 + psi_start + t] = psi1 - (psi0 + v0 * delta0 / Lf * dt);
      fg[1 + v_start + t] = v1 - (v0 + a0 * dt);
      fg[1 + cte_start + t] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
      fg[1 + epsi_start + t] = epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt);
    }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  // Set the number of constraints
  size_t n_const = state.size() * N;

  // Set the number of model variables
  size_t n_vars = n_const + 2 * (N - 1);

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }

  // define "vars lowerdounds" and "vars upperbounds" variables
  Dvector vars_lb(n_vars);
  Dvector vars_ub(n_vars);

  // Set bounds for all non-actuator variables
  for (i = 0; i < delta_start; i++) {
    vars_lb[i] = -100;
    vars_ub[i] = 100;
  }

  // Set lower / upper limits for delta actuator
  for (i = delta_start; i < a_start; i++) {
    vars_lb[i] = -0.436332;
    vars_ub[i] = 0.436332;
  }

  // Set lower / upper limits for acceleration actuator
  for (i = a_start; i < n_vars; i++) {
    vars_lb[i] = -0.2;
    vars_ub[i] = 0.2;
  }

  // Define "constraints lowerbounds" and "constraints upperbounds" variables
  Dvector const_lb(n_const);
  Dvector const_ub(n_const);

  // Should be 0 besides initial state.
  for (i = 0; i < n_const; i++) {
    const_lb[i] = 0;
    const_ub[i] = 0;
  }

  // set constraints from state
  const_lb[x_start] = const_ub[x_start] = state[0]; // x
  const_lb[y_start] = const_ub[y_start] = state[1]; // y
  const_lb[psi_start] = const_ub[psi_start] = state[2]; // psi
  const_lb[v_start] = const_ub[v_start] = state[3]; // v
  const_lb[cte_start] = const_ub[cte_start] = state[4]; // cte
  const_lb[epsi_start] = const_ub[epsi_start] = state[5]; // epsi

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lb, vars_ub, const_lb,
      const_ub, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // compose solution vector. First 2 elements are steering and throttle values. All other are trajectory points
  vector<double> result;

  // save steering and throttle as first elements in the result vector
  result.push_back(solution.x[delta_start]);
  result.push_back(solution.x[a_start]);

  // add points to draw the trajectory
  for (i = 0; i < N; i++) {
    result.push_back(solution.x[x_start + i]);
    result.push_back(solution.x[y_start + i]);
  }

  return result;
}
