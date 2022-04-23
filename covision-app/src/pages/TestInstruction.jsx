import React, { useEffect, useState } from 'react';
import { useParams, useHistory } from 'react-router-dom';
import instructions from '../data/instructions';
import './TestInstruction.css';
import { useTimer } from 'react-timer-hook';

const TestInstruction = () => {
  const { id } = useParams();
  const instruction = instructions[id];
  const numberOfSteps = instruction.steps.length;
  const testTime = instruction.time * 60;
  const [index, setIndex] = useState(0);
  const [displayedText, setDisplayedText] = useState('');
  const [timerRunning, setTimerRunning] = useState(false);
  let history = useHistory();
  const mainText = React.useRef();

  const { seconds, minutes, restart } = useTimer({
    testTime,
    onExpire: () => {
<<<<<<< HEAD
      console.log('timer expired');
      const text = 'The time of ' + instruction.time + ' is over. You will be redirected to the result detection.';
=======
      const text =
        'Die Zeit von ' + instruction.time + ' ist vorüber. Sie werden zur Ergebnis Erkennung weiter geleitet.';
>>>>>>> 7179c9ba2b0f3ed5b7460e812f833ef2b589cc00
      window.alert(text);
      history.push('/');
    },
  });

  useEffect(() => {
    mainText.current.focus();
    let text = instruction.steps[index];
    if (index === instruction.timerTriggerStep) {
      text = text + ' A timer of ' + instruction.time + ' minutes was set.';
    }
    setDisplayedText(text);
  }, [index, instruction.steps, instruction.time, instruction.timerTriggerStep, mainText]);

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
      <div className="subheader">
        Step {index + 1} out of {numberOfSteps}
      </div>
      <div className="mainText" ref={mainText}>
        {displayedText}
      </div>

      <div className="textFrame" />
      <div className="buttons">
        <button
          className="buttonPrev"
          onClick={() => {
            prevStep();
          }}
        >
          Back
        </button>
        <button
          className="buttonNext"
          onClick={() => {
            nextStep();
          }}
        >
          {index < 0 ? <>Start</> : <>Next</>}
        </button>
        <a href="/">
          <button className="buttonHome">Home</button>
        </a>
      </div>
      {timerRunning ? (
        <div className="timer">
          <span>{minutes}</span>:
          <span>
            {seconds.toLocaleString('en-US', {
              minimumIntegerDigits: 2,
              useGrouping: false,
            })}
          </span>
        </div>
      ) : (
        <></>
      )}
    </div>
  );
};

export default TestInstruction;
