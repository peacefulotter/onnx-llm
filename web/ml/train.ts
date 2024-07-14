import * as ort from 'onnxruntime-web/training'
import { SortData } from './data'

const SESSSION_PATH = './_next/static/chunks/pages/'

const getTrainingSession = async (): Promise<ort.TrainingSession> => {
    const createOptions: ort.TrainingSessionCreateOptions = {
        checkpointState: 'checkpoint',
        trainModel: 'training_model.onnx',
        evalModel: 'eval_model.onnx',
        optimizerModel: 'optimizer_model.onnx',
    }

    for (const key in createOptions) {
        createOptions[key as keyof typeof createOptions] = `${SESSSION_PATH}${
            createOptions[key as keyof typeof createOptions]
        }`
    }

    console.log('Creating training session with options: ', createOptions)

    try {
        const session = await ort.TrainingSession.create(createOptions, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
        })
        return session
    } catch (err) {
        console.log('Error loading the training session: ' + err)
        throw err
    }
}

const computeAccuracy = (
    batch: Record<string, ort.Tensor>,
    results: Record<string, ort.Tensor>,
    blockSize: number,
    vocabSize: number,
    batchSize: number
) => {
    const logits = results['output'].data as Float32Array
    const targets = batch['labels']

    const mid = Math.floor(blockSize / 2)

    let correct = 0
    for (let i = 0; i < batchSize; i++) {
        const sampleIndex = i * vocabSize * blockSize

        for (let j = 0; j <= mid; j++) {
            const index = sampleIndex + (j + mid) * vocabSize
            const values = logits.slice(index, index + vocabSize)

            let argmax = 0
            for (let k = 1; k < vocabSize; k++) {
                if (values[k] > values[argmax]) argmax = k
            }

            if (argmax == targets.data[i * blockSize + j + mid]) {
                correct++
            }
        }
    }

    return correct / (batchSize * (mid + 1))
}

export async function runTraining(
    maxIters: number = 1000,
    blockSize: number = 11,
    vocabSize: number = 3,
    batchSize: number = 16
) {
    const session = await getTrainingSession()
    const dataset = new SortData(session, blockSize, vocabSize, batchSize)
    let iter = 0

    const lossName = session.trainingOutputNames[0]
    const data = []

    while (iter < maxIters) {
        const start = Date.now()

        const batch = dataset.getTrainingBatch()
        const results = await session.runTrainStep(batch)

        const dt = Date.now() - start
        const loss = (results[lossName].data as Float32Array)[0]

        if (iter % 25 === 0) {
            console.log(`Training iteration ${iter} took ${dt}ms, loss: ${loss}`)
        }

        const acc = computeAccuracy(batch, results, blockSize, vocabSize, batchSize)
        data.push({ iter, loss, dt: dt / 1000, acc })

        // update weights then reset gradients
        await session.runOptimizerStep()
        await session.lazyResetGrad()

        ++iter
    }

    console.log({
        config: {
            maxIters,
            blockSize,
            vocabSize,
            batchSize,
        },
        data,
    })

    await session.release()
}
