import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import instructions from '../data/instructions';
import { IonText } from '@ionic/react';
import Speech from 'speak-tts';
import './TestInstruction.css';

const TestInstruction = () => {
  const { id } = useParams();
  const instruction = instructions[id];
  const numberOfSteps = instruction.steps.length;
  const [index, setIndex] = useState(-1);
  const [buttonsDisabled, setButtonsDisabled] = useState(false);

  useEffect(() => {
    triggerInstruction();
  }, [index]);

  const triggerInstruction = () => {
    const speech = new Speech();
    if (speech.hasBrowserSupport()) {
      console.log('speech synthesis supported');

      speech
        .init()
        .then((data) => {
          // The "data" object contains the list of available voices and the voice synthesis params
          console.log('Speech is ready, voices are available', data);

          speech
            .speak({
              text: instruction.steps[index],
              listeners: {
                onstart: () => {
                  setButtonsDisabled(true);
                },
                onend: () => {
                  setButtonsDisabled(false);
                },
              },
            })
            .then(() => {
              console.log('Success !');
            })
            .catch((e) => {
              console.error('An error occurred :', e);
            });
        })
        .catch((e) => {
          console.error('An error occured while initializing : ', e);
        });
    }
  };

  const nextStep = () => {
    if (index < numberOfSteps - 1) {
      setIndex(index + 1);
    } else {
      setIndex(index);
    }
  };

  const prevStep = () => {
    if (index >= 1) {
      setIndex(index - 1);
    } else {
      setIndex(index);
    }
  };

  return (
    <div className="pageContainer">
      <div className="header">
        <IonText>Instruction to rapid test with id {instruction.id}</IonText>
      </div>
      <div className="subheader">
        <IonText>
          {' '}
          Schritt {index + 1} von {numberOfSteps}
        </IonText>
      </div>
      <div className="buttons">
        <button
          className="buttonPrev"
          onClick={() => {
            prevStep();
          }}
          disabled={buttonsDisabled}
        >
          Vorheriger Schritt
        </button>
        <button
          className="buttonNext"
          onClick={() => {
            nextStep();
          }}
          disabled={buttonsDisabled}
        >
          {index < 0 ? <>Start</> : <>Nächster Schritt</>}
        </button>
      </div>
    </div>
  );
};

export default TestInstruction;
