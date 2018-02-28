/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

/*
 * For comparing distances, distance squared is faster
 */
inline double dist_sq(double x1, double y1, double x2, double y2) 
{
	return ((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
}

/*
 * Ensure yaw angle in range [0, 2*pi]
 */
inline double normalize_range(double theta)
{
	while (theta < 0.0)
		theta += 2*M_PI;
	while (theta > 2*M_PI)
		theta -= 2*M_PI;
	return theta;
}


void ParticleFilter::init(double x, double y, double theta, double std[])
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	particles.resize(num_particles);
	for (auto i = 0; i < num_particles; ++i)
	{
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = normalize_range(dist_theta(gen)); 
		particles[i].weight = 1.0;
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	for (auto i = 0; i < num_particles; ++i)
	{
		// noiseless prediction
		double theta_0 = particles[i].theta;
		double x = particles[i].x + velocity/yaw_rate*(sin(theta_0 + yaw_rate*delta_t) - sin(theta_0));
		double y = particles[i].y + velocity/yaw_rate*(cos(theta_0) - cos(theta_0 + yaw_rate*delta_t));
		double theta = theta_0 + yaw_rate*delta_t;

		// add noise - consider caching noise generator with zero mean
		normal_distribution<double> dist_x(x, std_pos[0]);
		normal_distribution<double> dist_y(y, std_pos[1]);
		normal_distribution<double> dist_theta(theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = normalize_range(dist_theta(gen));
	}
	
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

}

int ParticleFilter::nearestLandmark(double x, double y, const Map &m)
{
	// return the index in list of the nearest landmark in map to location (x,y)

	int nearest = 0;
	double min = dist_sq(x, y, m.landmark_list[0].x_f, m.landmark_list[0].y_f);
	for (auto i = 1; i < m.landmark_list.size(); ++i)
	{
		double current = dist_sq(x, y, m.landmark_list[i].x_f, m.landmark_list[i].y_f);
		if (current < min)
		{
			min = current;
			nearest = i;
		}
	}

	return nearest;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
								   const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	const int num_observations = observations.size();
	weights.clear();

	// for each particle
	for (auto p = particles.begin(); p != particles.end(); ++p)
	{
		p->sense_x.resize(num_observations);
		p->sense_y.resize(num_observations);
		p->associations.resize(num_observations);

		const double sig_x = std_landmark[0];
		const double sig_y = std_landmark[1];
		const double gauss_norm = 1.0/(2*M_PI*sig_x*sig_y);

		double x_p = p->x;
		double y_p = p->y;
		double theta_p = p->theta;

		p->weight = 1.0; // initially

		// for each observation
		for (auto i = 0; i < num_observations; ++i)
		{
			// transform observations to map coordinates from particle frame of reference
			double x_m = x_p + cos(theta_p)*observations[i].x - sin(theta_p)*observations[i].y;
			double y_m = y_p + sin(theta_p)*observations[i].x + cos(theta_p)*observations[i].y;
			p->sense_x[i] = x_m;
			p->sense_y[i] = y_m;

			// associate observation (x_m, y_m) with nearest landmark index from map
			int index = nearestLandmark(x_m, y_m, map_landmarks);
			p->associations[i] = map_landmarks.landmark_list[index].id_i;
			double dx = x_m - map_landmarks.landmark_list[index].x_f;
			double dy = y_m - map_landmarks.landmark_list[index].y_f;			

			// bi-variate Gaussian 
			double exponent = (dx*dx)/(2*sig_x*sig_x) + (dy*dy)/(2*sig_y*sig_y);
			double obs_weight = gauss_norm*exp(-exponent);
			p->weight *= obs_weight;
		}
		weights.push_back(p->weight);
	}
}

void ParticleFilter::resample()
{
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> d(weights.begin(), weights.end());

	std::vector<Particle> replacements;

	for (int i = 0; i < num_particles; ++i)
	{
		int index = d(gen); // index of particle selected by weighted random selection
		replacements.push_back(particles[index]);
	}
	// swap previous particle list with replacement
	particles = replacements;
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
										 const std::vector<double> &sense_x, const std::vector<double> &sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates
	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
