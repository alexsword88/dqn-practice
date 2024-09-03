# 環境設定

## 必要套件

### 安裝 Poetry

#### Ubuntu 23.04 or above

```shell
sudo apt update && sudo apt install pipx
pipx ensurepath
pipx install poetry
```

#### Ubuntu 22.04 or below

```shell
python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install poetry
```

#### Fedora

```shell
sudo dnf install pipx
pipx ensurepath
sudo pipx ensurepath --global # optional to allow pipx actions with --global argument
pipx install poetry
```

#### macOS

```shell
brew install pipx
pipx ensurepath
sudo pipx ensurepath --global # optional to allow pipx actions with --global argument
pipx install poetry
```

#### Windows

```powershell
python -m pip install --user pipx
pipx ensurepath
pipx install poetry
```

#### pipx 其他安裝方式

- https://github.com/pypa/pipx?tab=readme-ov-file#install-pipx

#### poetry 其他安裝方式

- https://python-poetry.org/docs/#installation

### 安裝 Taskfile

#### Linux / macOS

```shell
sudo sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin
```

#### Windows

```powershell
winget install Task.Task
```

#### 其他安裝方式

- https://taskfile.dev/installation/

## 建立環境

```shell
poetry config virtualenvs.in-project true
poetry install
cp .env.example .env
```

## 執行

### 訓練模型

```shell
task train
```

### 執行測試

```shell
task play
```

# 參考

- https://github.com/gordicaleksa/pytorch-learn-reinforcement-learning/blob/main/train_DQN_script.py
- https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
