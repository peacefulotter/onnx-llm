# Requirements
- Node.js <b>v16</b> (Tested on 16.20.0), use [nvm](https://github.com/nvm-sh/nvm) to install and manage Node.js versions
- NPM (Tested on 8.19.4)
- <it>Optional</it> If you use Bun, it should work too (Tested on 1.1.20)

# Installation

```sh
# 1. Make sure you have Node.js 16
node --version
# 1.2 If you have nvm just do:
nvm use 16

# 2. Install the required packages
npm install

# 3. Start the development server
npm run dev
```

## Export results to wandb

At the end of training, the recorded logs are logged in the console. Copy the object and paste it into a `training.json` file. Then run the following command:   
```sh
# Make sure you have "wandb" installed  
python export-wandb.py
```

