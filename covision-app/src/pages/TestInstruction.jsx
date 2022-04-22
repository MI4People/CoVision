import { useState } from "react"
import { useParams } from "react-router-dom";
import TextToSpeech from "../components/CovInstruction/TextToSpeech";
import instructions from "../data/instructions";
import {
    IonText,
    IonButton
  } from "@ionic/react";

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
            <IonText>Instruction to rapid test with id {instruction.id}</IonText>
            <IonText>{" "}Step {index+1} out of {numberOfSteps}</IonText>
            <TextToSpeech text={instruction.steps[index]}/>
            <IonButton onClick={() => nextStep()}>Next Step</IonButton>
        </>
    )
}

export default TestInstruction