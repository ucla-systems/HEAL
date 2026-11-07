#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_DIR="${VLLM_DIR:-${ROOT_DIR}/vllm}"
PATCH_DIR="${PATCH_DIR:-${ROOT_DIR}/patches}"
PRESET="${PRESET:-release}"
TARGET="${TARGET:-install}"
BUILD_DIR="${BUILD_DIR:-${VLLM_DIR}/cmake-build-release}"
PYTHON="${PYTHON:-python}"
TORCH_BACKEND="${TORCH_BACKEND:-cu128}"
UV_SYSTEM_PYTHON="${UV_SYSTEM_PYTHON:-1}"
CONFIRM_SETUP_OR_CONFIGURE="${CONFIRM_SETUP_OR_CONFIGURE:-0}"
ACTIVE_PATCH_FILE="${ACTIVE_PATCH_FILE:-${ROOT_DIR}/.vllm_active_patch}"
ACTIVE_PATCH_COPY="${ACTIVE_PATCH_COPY:-${ROOT_DIR}/.vllm_active_patch.diff}"

VERSION_NAMES=(
  VLLM_ORIGINAL
  VLLM_LAYERCAST
  VLLM_SIGNIFICAND_BITS
  VLLM_HEAL_BASE
  VLLM_HEAL_PLUS
  VLLM_HEAL_FULL
  VLLM_HEAL
  VLLM_DUMP_TENSORS
  VLLM_QKV_DISTRIBUTION
  VLLM_ERROR_CONTRIBUTION
  VLLM_ERROR_CONTRIBUTION_HEAL
  VLLM_SUMMARIZED
)

usage() {
  cat <<'EOF'
Usage:
  scripts/vllm_incremental_build.sh setup --confirm-setup-or-configure
  scripts/vllm_incremental_build.sh configure --confirm-setup-or-configure
  scripts/vllm_incremental_build.sh build
  scripts/vllm_incremental_build.sh all --confirm-setup-or-configure

  scripts/vllm_incremental_build.sh list-versions
  scripts/vllm_incremental_build.sh status
  scripts/vllm_incremental_build.sh save-version VERSION
  scripts/vllm_incremental_build.sh apply-version VERSION
  scripts/vllm_incremental_build.sh prepare-version VERSION
  scripts/vllm_incremental_build.sh revert-version [VERSION]

Environment:
  VLLM_DIR=<repo>/vllm             vLLM source directory
  PATCH_DIR=<repo>/patches         Canonical patch directory
  PRESET=release                    CMake preset name
  TARGET=install                    CMake target
  PYTHON=python                     Python used for CMake preset generation
  TORCH_BACKEND=cu128                uv PyTorch backend for CUDA dependencies
  UV_SYSTEM_PYTHON=1                 Pass --system to uv pip install
  JOBS=32                           Optional build parallelism
  CONFIRM_SETUP_OR_CONFIGURE=1      Alternative to the confirmation flag

Workflow:
  setup            Install CUDA/build requirements and a Python-only editable
                   vLLM package. This skips both precompiled wheel fetching and
                   full native source compilation.
  configure        Generate CMake presets and configure the release build tree.
  build            Incrementally rebuild and install changed native components.
  all              Run setup, configure, then incremental build.

  save-version     Save the current vLLM diff to patches/VERSION.patch.
                   This includes untracked files via an intent-to-add snapshot.
  apply-version    Apply patches/VERSION.patch to a clean vLLM worktree.
  prepare-version  Apply VERSION to a clean vLLM worktree, then build.
                   The patch stays active so experiments run patched code.
  revert-version   Reverse the active patch after the experiment run.

Recommended experiment loop:
  scripts/vllm_incremental_build.sh prepare-version VLLM_ORIGINAL
  # run the experiment while the patch is active
  scripts/vllm_incremental_build.sh revert-version

WARNING:
  Full rebuilds are intentionally not exposed here. Do not delete
  vllm/cmake-build-release or run a full source install unless the user
  explicitly asks for it. Normal kernel iteration should use "build".

  The setup/configure/all actions are guarded because they mutate the editable
  install or CMake build configuration. Before running them, confirm with the
  user that this non-incremental setup/configure step is actually needed, then
  pass --confirm-setup-or-configure.

  Applying a version requires a clean vLLM worktree. If code edits need to be
  preserved first, run save-version VERSION, then return vLLM to the clean base
  before applying another version.
EOF
}

ACTION="${1:-build}"
if [[ $# -gt 0 ]]; then
  shift
fi

POSITIONAL=()
for arg in "$@"; do
  case "${arg}" in
    --confirm-setup-or-configure)
      CONFIRM_SETUP_OR_CONFIGURE=1
      ;;
    -h|--help|help)
      ACTION="help"
      ;;
    *)
      POSITIONAL+=("${arg}")
      ;;
  esac
done

if [[ "${#POSITIONAL[@]}" -gt 1 ]]; then
  echo "Too many positional arguments: ${POSITIONAL[*]}" >&2
  echo >&2
  usage >&2
  exit 2
fi

VERSION="${POSITIONAL[0]:-}"

ensure_vllm_repo() {
  if ! git -C "${VLLM_DIR}" rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "Not a git repository: ${VLLM_DIR}" >&2
    exit 2
  fi
}

version_is_known() {
  local version="$1"
  local known
  for known in "${VERSION_NAMES[@]}"; do
    if [[ "${version}" == "${known}" ]]; then
      return 0
    fi
  done
  return 1
}

require_version() {
  if [[ -z "${VERSION}" ]]; then
    echo "VERSION is required for action '${ACTION}'." >&2
    echo "Use list-versions to see valid names." >&2
    exit 2
  fi
  if ! version_is_known "${VERSION}"; then
    echo "Unknown VERSION: ${VERSION}" >&2
    echo "Use list-versions to see valid names." >&2
    exit 2
  fi
}

patch_file_for_version() {
  local version="$1"
  echo "${PATCH_DIR}/${version}.patch"
}

list_versions() {
  local version
  for version in "${VERSION_NAMES[@]}"; do
    echo "${version} $(patch_file_for_version "${version}")"
  done
}

vllm_status_short() {
  git -C "${VLLM_DIR}" status --porcelain
}

require_clean_vllm() {
  ensure_vllm_repo
  local status
  status="$(vllm_status_short)"
  if [[ -z "${status}" ]]; then
    return
  fi

  cat >&2 <<EOF
Refusing to apply a version because vLLM has code changes:

${status}

If these edits are the version implementation, save them first:
  scripts/vllm_incremental_build.sh save-version VERSION

Then return vLLM to the clean base before applying another version.
EOF
  exit 2
}

active_patch_exists() {
  [[ -f "${ACTIVE_PATCH_FILE}" ]]
}

read_active_patch() {
  ACTIVE_VERSION="$(sed -n '1p' "${ACTIVE_PATCH_FILE}")"
  ACTIVE_PATCH_SOURCE="$(sed -n '2p' "${ACTIVE_PATCH_FILE}")"
}

require_no_active_patch() {
  if ! active_patch_exists; then
    return
  fi
  read_active_patch
  cat >&2 <<EOF
An active vLLM patch is already recorded:
  ${ACTIVE_VERSION}
  ${ACTIVE_PATCH_SOURCE}

Run this after the experiment finishes:
  scripts/vllm_incremental_build.sh revert-version
EOF
  exit 2
}

require_setup_or_configure_confirmation() {
  if [[ "${CONFIRM_SETUP_OR_CONFIGURE}" == "1" ]]; then
    return
  fi

  cat >&2 <<'EOF'
Refusing to run setup/configure without explicit confirmation.

Normal vLLM kernel iteration should use:
  scripts/vllm_incremental_build.sh build

setup/configure/all can change the editable install or regenerate the CMake
build configuration. Confirm with the user first. If it is really needed, rerun
with:
  --confirm-setup-or-configure

If you think a true full source rebuild is needed, stop and ask the user
explicitly; this script intentionally does not perform full rebuild cleanup or
full source reinstall.
EOF
  exit 2
}

uv_pip_install() {
  local uv_args=(pip install)
  if [[ "${UV_SYSTEM_PYTHON}" == "1" ]]; then
    uv_args+=(--system)
  fi
  uv "${uv_args[@]}" "$@"
}
run_setup() {
  cd "${VLLM_DIR}"
  unset VLLM_USE_PRECOMPILED
  unset VLLM_PRECOMPILED_WHEEL_LOCATION
  unset VLLM_PRECOMPILED_WHEEL_COMMIT
  unset VLLM_PRECOMPILED_WHEEL_VARIANT

  uv_pip_install \
    -r requirements/build.txt \
    -r requirements/cuda.txt \
    --torch-backend="${TORCH_BACKEND}"

  VLLM_TARGET_DEVICE=empty uv_pip_install \
    -e . \
    --no-build-isolation \
    --torch-backend="${TORCH_BACKEND}"
}

run_configure() {
  cd "${VLLM_DIR}"
  "${PYTHON}" tools/generate_cmake_presets.py --force-overwrite
  cmake --preset "${PRESET}"
}

run_build() {
  cd "${VLLM_DIR}"
  local build_args
  if [[ -f "CMakePresets.json" || -f "CMakeUserPresets.json" ]]; then
    build_args=(--build --preset "${PRESET}" --target "${TARGET}")
  elif [[ -f "${BUILD_DIR}/build.ninja" || -f "${BUILD_DIR}/Makefile" ]]; then
    build_args=(--build "${BUILD_DIR}" --target "${TARGET}")
  else
    build_args=(--build --preset "${PRESET}" --target "${TARGET}")
  fi
  if [[ -n "${JOBS:-}" ]]; then
    build_args+=(-j "${JOBS}")
  fi
  cmake "${build_args[@]}"
}

save_version() {
  require_version

  ensure_vllm_repo
  mkdir -p "${PATCH_DIR}"

  local patch_file
  patch_file="$(patch_file_for_version "${VERSION}")"

  mapfile -t untracked < <(git -C "${VLLM_DIR}" ls-files --others --exclude-standard)
  if [[ "${#untracked[@]}" -gt 0 ]]; then
    git -C "${VLLM_DIR}" add -N -- "${untracked[@]}"
  fi

  git -C "${VLLM_DIR}" diff --binary -- . > "${patch_file}"

  if [[ "${#untracked[@]}" -gt 0 ]]; then
    git -C "${VLLM_DIR}" reset -q -- "${untracked[@]}"
  fi

  if [[ ! -s "${patch_file}" ]]; then
    rm -f "${patch_file}"
    echo "No vLLM diff to save for ${VERSION}." >&2
    exit 2
  fi

  echo "Saved ${VERSION} to ${patch_file}"
  sha256sum "${patch_file}"
}

apply_version() {
  require_version
  require_no_active_patch
  require_clean_vllm

  local patch_file
  patch_file="$(patch_file_for_version "${VERSION}")"
  if [[ ! -f "${patch_file}" ]]; then
    echo "Missing patch file for ${VERSION}: ${patch_file}" >&2
    exit 2
  fi

  git -C "${VLLM_DIR}" apply --check "${patch_file}"
  git -C "${VLLM_DIR}" apply "${patch_file}"
  cp "${patch_file}" "${ACTIVE_PATCH_COPY}"
  printf '%s\n%s\n' "${VERSION}" "${patch_file}" > "${ACTIVE_PATCH_FILE}"
  echo "Applied ${VERSION}"
}

prepare_version() {
  apply_version
  if ! run_build; then
    cat >&2 <<EOF
Build failed while ${VERSION} is active.
The patch is still applied so you can inspect or fix it. When done, run:
  scripts/vllm_incremental_build.sh revert-version
EOF
    return 1
  fi
  cat <<EOF
${VERSION} is built and still active for the experiment run.
After the experiment finishes, run:
  scripts/vllm_incremental_build.sh revert-version
EOF
}

resolve_revert_target() {
  if active_patch_exists; then
    read_active_patch
    if [[ -n "${VERSION}" && "${VERSION}" != "${ACTIVE_VERSION}" ]]; then
      cat >&2 <<EOF
Active patch is ${ACTIVE_VERSION}, but requested revert is ${VERSION}.
Run without VERSION to revert the active patch.
EOF
      exit 2
    fi
    VERSION="${ACTIVE_VERSION}"
    REVERT_PATCH="${ACTIVE_PATCH_COPY}"
    if [[ ! -f "${REVERT_PATCH}" ]]; then
      REVERT_PATCH="${ACTIVE_PATCH_SOURCE}"
    fi
    return
  fi

  if [[ -z "${VERSION}" ]]; then
    echo "No active patch is recorded; nothing to revert." >&2
    echo "To reverse a specific patch explicitly, pass VERSION." >&2
    exit 2
  fi
  require_version
  REVERT_PATCH="$(patch_file_for_version "${VERSION}")"
}

revert_version() {
  resolve_revert_target

  if [[ ! -f "${REVERT_PATCH}" ]]; then
    echo "Missing patch file to revert ${VERSION}: ${REVERT_PATCH}" >&2
    exit 2
  fi

  git -C "${VLLM_DIR}" apply -R --check "${REVERT_PATCH}"
  git -C "${VLLM_DIR}" apply -R "${REVERT_PATCH}"

  if active_patch_exists; then
    rm -f "${ACTIVE_PATCH_FILE}" "${ACTIVE_PATCH_COPY}"
  fi

  local status
  status="$(vllm_status_short)"
  if [[ -n "${status}" ]]; then
    cat >&2 <<EOF
Reverted ${VERSION}, but vLLM is not clean:

${status}
EOF
    exit 1
  fi
  echo "Reverted ${VERSION}; vLLM is clean."
}

show_status() {
  ensure_vllm_repo
  if active_patch_exists; then
    read_active_patch
    echo "Active patch: ${ACTIVE_VERSION}"
    echo "Patch source: ${ACTIVE_PATCH_SOURCE}"
  else
    echo "Active patch: none"
  fi
  echo
  git -C "${VLLM_DIR}" status --short
}

case "${ACTION}" in
  setup)
    require_setup_or_configure_confirmation
    run_setup
    ;;
  configure)
    require_setup_or_configure_confirmation
    run_configure
    ;;
  build)
    run_build
    ;;
  all)
    require_setup_or_configure_confirmation
    run_setup
    run_configure
    run_build
    ;;
  list-versions)
    list_versions
    ;;
  status)
    show_status
    ;;
  save-version)
    save_version
    ;;
  apply-version)
    apply_version
    ;;
  prepare-version)
    prepare_version
    ;;
  revert-version)
    revert_version
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
