import React from 'react';

interface DonationButtonProps {
  size?: 'small' | 'medium' | 'large';
  text?: string;
  variant?: 'primary' | 'ghost' | 'inline';
  context?: string;
  className?: string;
}

const DonationButton: React.FC<DonationButtonProps> = ({ 
  size = 'medium', 
  text = 'Support This Project',
  variant = 'primary',
  context = '',
  className = ''
}) => {
  const getSizeClasses = () => {
    switch (size) {
      case 'small': return 'text-sm';
      case 'large': return 'text-lg';
      default: return 'text-base';
    }
  };

  // Context-specific messages for our devious plan
  const getContextMessage = () => {
    const messages: Record<string, string> = {
      'victory': '🎉 Celebrate your win by supporting the project!',
      'defeat': '🤖 Help the AI get even stronger!',
      'thinking': '💭 This analysis costs ~$0.05 in GPU time',
      'stats': '📊 Your games help improve the AI',
      'community': '👥 Keep community games free for everyone',
      'slow': '⚡ Help us get faster servers!',
      'training': '🧠 Support continued AI training'
    };
    return messages[context] || text;
  };

  // Get button text based on context for maximum psychological impact
  const getButtonText = () => {
    const contextButtons: Record<string, string> = {
      'thinking': 'Buy the AI a coffee ☕',
      'victory': 'Celebrate with coffee! ☕',
      'defeat': 'Help AI improve ☕',
      'stats': 'Support this project ☕',
      'community': 'Keep it free ☕',
      'slow': 'Upgrade servers ☕',
      'training': 'Fund training ☕'
    };
    return contextButtons[context] || text;
  };

  return (
    <div className={`donation-section ${className}`}>
      {/* PayPal styles */}
      <style>{`
        .pp-TMAYM2RD56756 {
          text-align: center;
          border: none;
          border-radius: 0.25rem;
          min-width: 11.625rem;
          padding: 0 2rem;
          height: 2.625rem;
          font-weight: bold;
          background-color: #FFD140;
          color: #000000;
          font-family: "Helvetica Neue", Arial, sans-serif;
          font-size: 1rem;
          line-height: 1.25rem;
          cursor: pointer;
          transition: all 0.2s ease;
        }
        .pp-TMAYM2RD56756:hover {
          background-color: #FFC107;
          transform: translateY(-1px);
          box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .pp-TMAYM2RD56756:active {
          transform: translateY(0px);
        }
        .donation-form-container {
          display: inline-grid;
          justify-items: center;
          align-content: start;
          gap: 0.5rem;
        }
      `}</style>

      {/* Context-specific psychological messaging */}
      {context && (
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-3 text-center">
          {getContextMessage()}
        </p>
      )}

      <div className="flex flex-col items-center gap-2">
        {/* Simple clickable text link for non-contextual buttons */}
        {!context && variant !== 'inline' && (
          <a 
            href="https://www.paypal.com/ncp/payment/TMAYM2RD56756"
            target="_blank"
            rel="noopener noreferrer"
            className={`font-medium ${getSizeClasses()} text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors cursor-pointer`}
          >
            {text}
          </a>
        )}

        {/* For contextual buttons, show the PayPal form */}
        {context && (
          <form 
            action="https://www.paypal.com/ncp/payment/TMAYM2RD56756" 
            method="post" 
            target="_blank" 
            className="donation-form-container"
          >
            <input 
              className="pp-TMAYM2RD56756" 
              type="submit" 
              value={getButtonText()}
            />
          </form>
        )}

        {/* Extra psychological nudge for high-engagement contexts */}
        {(context === 'thinking' || context === 'victory') && (
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 text-center max-w-xs">
            Your support helps keep this chess AI free and improving! 🚀
          </p>
        )}
      </div>
    </div>
  );
};

export default DonationButton; 