import { useState } from "react"
import { useParams } from "react-router-dom";
import TextToSpeech from "../components/CovInstruction/TextToSpeech";
import instructions from "../data/instructions";
import {
    IonText,
    IonButton
  } from "@ionic/react";
import "./TestInstruction.css"

const TestInstruction = () => {
    const { id } = useParams();
    const instruction = instructions[id]
    const numberOfSteps = instruction.steps.length
    const [index, setIndex] = useState(0)

    const nextStep = () => {
        setIndex(index+1)
    }

    return (
        <>
            <div className="header">
                <IonText>Instruction to rapid test with id {instruction.id}</IonText>
            </div>
            <div className="subheader">
                <IonText>{" "}Step {index+1} out of {numberOfSteps}</IonText>
            </div>
            <div className="playButton">
                <TextToSpeech text={instruction.steps[index]} step={index+1}/>
            </div>
            <div className="nextButton">
                <IonButton onClick={() => nextStep()}>Next Step</IonButton>
            </div>
        </>
    )
}

export default TestInstruction