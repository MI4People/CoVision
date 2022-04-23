import { useEffect, useState } from 'react';
import { useParams, useHistory } from 'react-router-dom';
import instructions from '../data/instructions';
import Speech from 'speak-tts';
import './TestInstruction.css';
import { useTimer } from 'react-timer-hook';

const TestInstruction = () => {
  const { id } = useParams();
  const instruction = instructions[id];
  const numberOfSteps = instruction.steps.length;
  const testTime = instruction.time * 60;
  const [index, setIndex] = useState(10);
  const [displayedText, setDisplayedText] = useState('');
  const [buttonsDisabled, setButtonsDisabled] = useState(false);
  const [timerRunning, setTimerRunning] = useState(false);
  let history = useHistory();

  const { seconds, minutes, restart } = useTimer({
    testTime,
    onExpire: () => {
      console.log('timer expired');
      const text =
        'Die Zeit von ' + instruction.time + ' ist vorüber. Sie werden zur Ergebnis Erkennung weiter geleitet.';
      triggerInstruction(text);
      history.push('/');
    },
  });

  useEffect(() => {
    let text = instruction.steps[index];
    if (index === instruction.timerTriggerStep) {
      text = text + 'Ein timer von ' + instruction.time + ' Minuten wurde gesetzt.';
    }
    setDisplayedText(text);
    // triggerInstruction(text);
  }, [index, instruction.steps, instruction.time, instruction.timerTriggerStep]);

  const triggerInstruction = (text) => {
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
              text: text,
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

  const startTimer = () => {
    setTimerRunning(true);
    const time = new Date();
    time.setSeconds(time.getSeconds() + testTime);
    restart(time);
  };

  const nextStep = () => {
    if (index < numberOfSteps - 1) {
      if (index + 1 === instruction.timerTriggerStep) {
        startTimer();
      }
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
      {/* <div className="header">
        <IonText>Anleitung für Test {instruction.id}</IonText>
      </div> */}
      <div className="subheader">
        Schritt {index + 1} von {numberOfSteps}
      </div>
      <div className="mainText">{displayedText}</div>
      <div className="textFrame" />
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
      {timerRunning ? (
        <div className="timer">
          <span>{minutes}</span>:<span>{seconds}</span>
        </div>
      ) : (
        <></>
      )}
    </div>
  );
};

export default TestInstruction;
