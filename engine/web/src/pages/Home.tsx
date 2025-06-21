import React from 'react';
import { Link } from 'react-router-dom';

const Home: React.FC = () => {
  return (
    <div className="pt-20 min-h-screen bg-gray-900 text-white">
      <div className="absolute left-1/2 transform -translate-x-1/2 w-full max-w-7xl px-8">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h1 className="text-6xl font-bold mb-8">Welcome to Rival Chess</h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Help train and challenge a state-of-the-art Graph Neural Network chess engine 
            through gameplay and collaborative decision making.
          </p>
        </div>

        {/* Disclaimer Banner */}
        <div className="mb-16 max-w-5xl mx-auto">
          <div className="bg-gradient-to-r from-red-900 to-orange-900 border-2 border-red-600 rounded-xl p-8 text-center">
            <div className="flex justify-center items-center mb-4">
              <svg className="w-8 h-8 text-yellow-300 mr-3" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <h2 className="text-2xl font-bold text-yellow-300">⚠️ EXPERIMENTAL SOFTWARE ⚠️</h2>
            </div>
            <div className="text-white text-lg leading-relaxed space-y-3">
              <p className="font-semibold">
                This is a work-in-progress research project with <span className="text-yellow-300 font-bold">NO GUARANTEES</span> of uptime or functionality.
              </p>
              <p>
                • The AI model is in active development and may behave unpredictably<br/>
                • Server may go down at any time without notice<br/>
                • Features may break, change, or disappear entirely<br/>
                • No user data persistence is guaranteed
              </p>
            </div>
          </div>
        </div>

        {/* Game Modes */}
        <div className="grid md:grid-cols-2 gap-12 mb-16 max-w-5xl mx-auto">
          {/* Training Mode */}
          <div className="bg-gray-800 rounded-xl p-8 text-center transform transition-transform hover:scale-105">
            <h2 className="text-2xl font-bold mb-4">Train the Model</h2>
            <p className="text-gray-300 mb-8 text-lg">
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
          <div className="bg-gray-800 rounded-xl p-8 text-center transform transition-transform hover:scale-105">
            <h2 className="text-2xl font-bold mb-4">Community Challenge</h2>
            <p className="text-gray-300 mb-8 text-lg">
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
        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          <div className="bg-gray-800 rounded-xl p-8 text-center transform transition-transform hover:scale-105">
            <h3 className="text-xl font-bold mb-4">Graph Neural Network</h3>
            <p className="text-gray-300 text-lg leading-relaxed">
              Experience chess against a modern AI that uses graph neural networks to understand chess positions in a way similar to humans.
            </p>
          </div>
          <div className="bg-gray-800 rounded-xl p-8 text-center transform transition-transform hover:scale-105">
            <h3 className="text-xl font-bold mb-4">Continuous Learning</h3>
            <p className="text-gray-300 text-lg leading-relaxed">
              The model learns from every game played, continuously improving its understanding and adapting to different playing styles.
            </p>
          </div>
          <div className="bg-gray-800 rounded-xl p-8 text-center transform transition-transform hover:scale-105">
            <h3 className="text-xl font-bold mb-4">Community Driven</h3>
            <p className="text-gray-300 text-lg leading-relaxed">
              Participate in a unique voting system where players collectively decide moves against the AI's strongest version.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home; 