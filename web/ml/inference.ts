import * as ort from 'onnxruntime-web'
import { SortData } from './data'

// Create session and set options. See the docs here for more options:
// https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#graphOptimizationLevel
const getSession = async (modelType: string, vocabSize: number, blockSize: number) => {
    const path = `${modelType}_vs=${vocabSize}_bs=${blockSize}.onnx`
    const session = await ort.InferenceSession.create(`./_next/static/chunks/pages/${path}`, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
    })
    console.log('Inference session created')
    return session
}

const getInferenceBatch = (
    session: ort.InferenceSession,
    blockSize: number,
    vocabSize: number,
    batchSize: number
) => {
    const data = new SortData(session, blockSize, vocabSize, batchSize)
    const tensor = data.getInferenceData()
    const feeds: Record<string, ort.Tensor> = {}
    feeds[session.inputNames[0]] = tensor
    return feeds
}

const formatOutput = (
    outputSoftmax: number[],
    vocabSize: number,
    blockSize: number,
    batchSize: number
) => {
    const res = []
    for (let i = 0; i < batchSize; i++) {
        let sample = []
        const sampleIndex = i * vocabSize * blockSize
        for (let j = 0; j < blockSize; j++) {
            const index = sampleIndex + j * vocabSize
            const values = outputSoftmax.slice(index, index + vocabSize)
            const argmax = values.indexOf(Math.max(...values))
            sample.push(argmax)
        }
        res.push(sample)
    }
}

export async function runInference(
    modelType: string = 'gpt-nano',
    vocabSize: number = 3,
    blockSize: number = 11,
    batchSize: number = 2
) {
    const session = await getSession(modelType, vocabSize, blockSize)
    const feeds = getInferenceBatch(session, blockSize, vocabSize, batchSize)

    // Run the session inference.
    const start = new Date()
    const outputData = await session.run(feeds)
    const end = new Date()
    const inferenceTime = (end.getTime() - start.getTime()) / 1000

    await session.release()

    // Get output results with the output name from the model export.
    const output = outputData[session.outputNames[0]]
    const outputSoftmax = softmax(Array.prototype.slice.call(output.data))
    const outputFomatted = formatOutput(outputSoftmax, vocabSize, blockSize, batchSize)

    const jsxInput = ['todo']

    return {
        input: jsxInput,
        res: outputFomatted,
        inferenceTime,
    }
}

//The softmax transforms values to be between 0 and 1
function softmax(resultArray: number[]): any {
    // Get the largest value in the array.
    const largestNumber = Math.max(...resultArray)
    // Apply exponential function to each result item subtracted by the largest number, use reduce to get the previous result number and the current number to sum all the exponentials results.
    const sumOfExp = resultArray
        .map((resultItem) => Math.exp(resultItem - largestNumber))
        .reduce((prevNumber, currentNumber) => prevNumber + currentNumber)
    //Normalizes the resultArray by dividing by the sum of all exponentials; this normalization ensures that the sum of the components of the output vector is 1.
    return resultArray.map((resultValue, index) => {
        return Math.exp(resultValue - largestNumber) / sumOfExp
    })
}
