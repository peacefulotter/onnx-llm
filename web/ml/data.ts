import * as ort from 'onnxruntime-web/training'

const isTrainingSession = (
    session: ort.TrainingSession | ort.InferenceSession
): session is ort.TrainingSession => {
    return 'trainingInputNames' in session
}

export class SortData {
    public inputName = ''
    public outputName = ''

    constructor(
        public session: ort.TrainingSession | ort.InferenceSession,
        public blockSize: number,
        public vocabSize: number,
        public batchSize: number
    ) {
        if (isTrainingSession(session)) {
            this.inputName = session.trainingInputNames[0]
            this.outputName = 'labels' // session.evalOutputNames[0]
        } else {
            this.inputName = session.inputNames[0]
            this.outputName = session.outputNames[0]
        }
        console.log('SortData', this.inputName, this.outputName)
    }

    private createRandomSample = (): number[] => {
        const sample = []
        for (let i = 0; i < this.blockSize; i++) {
            sample.push(Math.floor(Math.random() * this.vocabSize))
        }
        return sample
    }

    private toInputTensor(x: number[]): ort.Tensor {
        const tensor = new ort.Tensor('int32', Int32Array.from(x), [this.batchSize, this.blockSize])
        return tensor
    }

    private toOutputTensor(x: number[]): ort.Tensor {
        let y = BigInt64Array.from(Array.from(x).map((x) => BigInt(x)))
        const tensor = new ort.Tensor('int64', y, [this.batchSize * this.blockSize])
        return tensor
    }

    getInputBatch(): number[] {
        const arr: number[] = []
        for (let i = 0; i < this.batchSize; i++) {
            const sample = this.createRandomSample()
            arr.push(...sample)
        }
        return arr
    }

    getInferenceData = () => {
        const arr = this.getInputBatch()
        return this.toInputTensor(arr)
    }

    private convertToTarget(x: number[]): number[] {
        const y_arr: number[] = []

        for (let i = 0; i < this.batchSize; i++) {
            const sample = new Array(this.blockSize).fill(-1)
            const mid = Math.floor(this.blockSize / 2)
            const xSampledSorted = x.slice(i * this.blockSize, i * this.blockSize + mid + 1).sort()
            for (let j = mid; j < this.blockSize; j++) {
                sample[j] = xSampledSorted[j - mid]
            }
            y_arr.push(...sample)
        }

        return y_arr
    }

    public getTrainingBatch() {
        const x_arr = this.getInputBatch()
        const y_arr = this.convertToTarget(x_arr)

        const x = this.toInputTensor(x_arr)
        const y = this.toOutputTensor(y_arr)

        return {
            [this.inputName]: x,
            [this.outputName]: y,
        }
    }
}
