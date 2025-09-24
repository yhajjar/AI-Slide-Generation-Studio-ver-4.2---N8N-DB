import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
}

const Card: React.FC<CardProps> = ({ children, className = '', onClick }) => {
  const baseClasses =
    'bg-white border border-gray-200 rounded-lg p-6 transition-all duration-300 shadow-md';
  const interactiveClasses = onClick
    ? 'cursor-pointer hover:border-[#219ebc] hover:bg-gray-50 transform hover:-translate-y-1'
    : '';

  return (
    <div className={`${baseClasses} ${interactiveClasses} ${className}`} onClick={onClick}>
      {children}
    </div>
  );
};

export default Card;