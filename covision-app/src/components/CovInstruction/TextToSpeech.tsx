import React from "react";
import Speech from "react-speech";

interface TextToSpeechProps {
  text: any;
}

const TextToSpeech: React.FC<TextToSpeechProps> = ({text}) => {

  return <Speech voice="Anna" text={text} />;
};

export default TextToSpeech;
