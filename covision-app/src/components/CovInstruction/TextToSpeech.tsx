import React from "react";
import Speech from "react-speech";

interface TextToSpeechProps {
  text: any;
  step: any;
}

const TextToSpeech: React.FC<TextToSpeechProps> = ({text, step}) => {
  const displayText = "Read Step " + step
  return (
      <Speech textAsButton={true} displayText={displayText} voice="Anna" text={text} />
    );
};

export default TextToSpeech;
