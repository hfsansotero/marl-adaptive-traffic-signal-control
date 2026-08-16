#!/bin/bash
# Ejecuta experimentos de LibSignal en secuencia, leyendo una cola de runs.
#
# Uso:
#   bash run_experiments.sh [opciones]
#
# Opciones:
#   --queue  FICHERO   Fichero de cola (default: queue.txt junto a este script)
#   --gpu    ID        Índice de GPU, -1 para CPU (default: 0)
#   --prefix NOMBRE    Prefijo de experimento (default: exp01)
#   --parallel         Lanzar todas las runs en paralelo (default: secuencial)
#
# Formato de queue.txt:
#   Una run por línea: <agente> <dataset>
#   Líneas vacías y líneas que empiezan por # se ignoran.
#
# Ejemplo de queue.txt:
#   frap        jinan3x4
#   axlight     cityflow4x4
#   axlight     jinan3x4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBSIGNAL_DIR="$SCRIPT_DIR/../LibSignal"
RESULTS_DIR="$SCRIPT_DIR/../results"

# --- Defaults ---
QUEUE_FILE="$SCRIPT_DIR/queue.txt"
GPU=0
PREFIX="exp01"
PARALLEL=false

# --- Parseo de argumentos ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --queue)    QUEUE_FILE="$2"; shift 2 ;;
    --gpu)      GPU="$2";        shift 2 ;;
    --prefix)   PREFIX="$2";     shift 2 ;;
    --parallel) PARALLEL=true;   shift   ;;
    *) echo "Opción desconocida: $1"; exit 1 ;;
  esac
done

# --- Validaciones ---
if [[ ! -f "$QUEUE_FILE" ]]; then
  echo "ERROR: no se encuentra el fichero de cola: $QUEUE_FILE"
  echo "Crea $SCRIPT_DIR/queue.txt con una run por línea: <agente> <dataset>"
  exit 1
fi

mkdir -p "$RESULTS_DIR"
cd "$LIBSIGNAL_DIR" || { echo "ERROR: no se encuentra $LIBSIGNAL_DIR"; exit 1; }

# --- Leer cola (ignorar comentarios y líneas vacías) ---
mapfile -t RUNS < <(grep -v '^\s*#' "$QUEUE_FILE" | grep -v '^\s*$')

if [[ ${#RUNS[@]} -eq 0 ]]; then
  echo "ERROR: queue.txt está vacío o solo contiene comentarios."
  exit 1
fi

echo "══════════════════════════════════════════════════════"
echo "  LibSignal Experiment Runner"
echo "  GPU=$GPU  Prefix=$PREFIX  Parallel=$PARALLEL"
echo "  Cola: $QUEUE_FILE (${#RUNS[@]} runs)"
echo "  Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════"
for run in "${RUNS[@]}"; do
  echo "  · $run"
done
echo ""

run_one() {
  local AGENT DATASET
  read -r AGENT DATASET <<< "$1"
  local LOGFILE="$RESULTS_DIR/log_${AGENT}_${DATASET}.txt"

  echo "┌─ INICIO: $AGENT on $DATASET | $(date '+%H:%M:%S')"
  conda run -n tfm python run.py \
    -a "$AGENT" -w cityflow -n "$DATASET" \
    --ngpu "$GPU" --prefix "$PREFIX" \
    2>&1 | tee -a "$LOGFILE"
  echo "└─ FIN:   $AGENT on $DATASET | $(date '+%H:%M:%S')"
  echo ""
}

if [[ "$PARALLEL" == true ]]; then
  echo "Modo PARALELO — lanzando ${#RUNS[@]} runs simultáneas"
  PIDS=()
  for run in "${RUNS[@]}"; do
    run_one "$run" &
    PIDS+=($!)
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
else
  echo "Modo SECUENCIAL — ${#RUNS[@]} runs en orden"
  echo ""
  for run in "${RUNS[@]}"; do
    run_one "$run"
  done
fi

echo "══════════════════════════════════════════════════════"
echo "  Todos los experimentos completados"
echo "  Fin: $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════"
