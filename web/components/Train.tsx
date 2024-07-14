import type { NextPage } from 'next'
import { runTraining } from '../ml/train'

const Train: NextPage = () => {
    const _runTraining = async () => {
        await runTraining()
    }

    return (
        <div>
            <button onClick={_runTraining}>Train LLM</button>
        </div>
    )
}

export default Train
