import type { NextPage } from 'next'
import { useState } from 'react'
import { runLLM } from '../utils/inference'

const Inference: NextPage = () => {
    const [time, setTime] = useState(0)
    const [input, setInput] = useState('')
    const [result, setResult] = useState('')

    const runLLMInference = async () => {
        const { input, res, inferenceTime } = await runLLM()
        setTime(inferenceTime)
        setInput(JSON.stringify(input, null, 2))
        setResult(JSON.stringify(res, null, 2))
    }

    return (
        <div>
            <button onClick={runLLMInference}>Run LLM inference</button>
            <div>
                <p>Inference time: {time} seconds</p>
                <div style={{ display: 'flex', gap: '30px' }}>
                    <div>
                        <b>Input:</b>
                        <pre>{input}</pre>
                    </div>
                    <div>
                        <b>Output:</b>
                        <pre>{result}</pre>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Inference
