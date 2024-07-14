import type { NextPage } from 'next'
import { runTraining } from '../utils/train'

const Train: NextPage = () => {
    const runLLMTrain = async () => {
        await runTraining()
    }

    return (
        <div>
            <button onClick={runLLMTrain}>Train LLM</button>
        </div>
    )
}

export default Train
