import React from 'react';
import { Link } from 'react-router-dom';

const Home: React.FC = () => {
  return (
    <div className="pt-10 pb-3 bg-white dark:bg-gray-900 text-gray-900 dark:text-white transition-colors">
      <div className="relative max-w-7xl mx-auto px-8">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h1 className="text-6xl font-bold mb-6">Welcome to Rival Chess</h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Help train and challenge a state-of-the-art Graph Neural Network chess engine 
            through gameplay and collaborative decision making.
          </p>
        </div>

        {/* Game Modes */}
        <div className="grid md:grid-cols-2 gap-8 mb-10 max-w-5xl mx-auto">
          {/* Training Mode */}
          <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-6 text-center transform transition-all hover:scale-105">
            <h2 className="text-2xl font-bold mb-3 text-gray-900 dark:text-white">Train the Model</h2>
            <p className="text-gray-600 dark:text-gray-300 mb-6 text-lg">
              Play one-on-one against the AI and help improve its learning through your games. 
              Each game contributes to the model's training data, making it stronger over time.
            </p>
            <Link 
              to="/train" 
              className="inline-block bg-blue-600 hover:bg-blue-700 text-white font-bold py-4 px-8 rounded-lg transition-colors text-lg"
            >
              Start Training
            </Link>
          </div>

          {/* Community Mode */}
          <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-6 text-center transform transition-all hover:scale-105">
            <h2 className="text-2xl font-bold mb-3 text-gray-900 dark:text-white">Community Challenge</h2>
            <p className="text-gray-600 dark:text-gray-300 mb-6 text-lg">
              Join forces with other players to challenge Rival's strongest model. 
              Vote on moves and work together to defeat the AI in this unique collaborative format.
            </p>
            <Link 
              to="/community" 
              className="inline-block bg-green-600 hover:bg-green-700 text-white font-bold py-4 px-8 rounded-lg transition-colors text-lg"
            >
              Join Challenge
            </Link>
          </div>
        </div>

        {/* Features Section */}
        <div className="grid md:grid-cols-3 gap-6 max-w-6xl mx-auto">
          <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-6 text-center transform transition-all hover:scale-105">
            <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">Graph Neural Network</h3>
            <p className="text-gray-600 dark:text-gray-300 text-lg leading-relaxed">
              Experience chess against a modern AI that uses graph neural networks to understand chess positions in a way similar to humans.
            </p>
          </div>
          <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-6 text-center transform transition-all hover:scale-105">
            <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">Continuous Learning</h3>
            <p className="text-gray-600 dark:text-gray-300 text-lg leading-relaxed">
              The model learns from every game played, continuously improving its understanding and adapting to different playing styles.
            </p>
          </div>
          <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-6 text-center transform transition-all hover:scale-105">
            <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">Community Driven</h3>
            <p className="text-gray-600 dark:text-gray-300 text-lg leading-relaxed">
              Participate in a unique voting system where players collectively decide moves against the AI's strongest version.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home; 