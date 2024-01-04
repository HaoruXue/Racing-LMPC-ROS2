.ONESHELL:
SHELL := /bin/bash
.DEFAULT_GOAL := build

.PHONY: clean
clean:
	@rm -rf build/ install/ log/ logs/ tests/

.PHONY: purge
purge:
	@rm -rf build/ install/ log/ logs/ tests/ src/external

.PHONY: vcs-import
vcs-import:
	@VCS_FILE="${VCS_FILE}"
	vcs import < ${VCS_FILE}

.PHONY: build-debug
build-debug:
	@PACKAGES="${PACKAGES}"
	source ./tools/scripts/source_all.sh
	if [ -z "$${PACKAGES}" ] ; then
		colcon build --cmake-args -DCMAKE_BUILD_TYPE=Debug --packages-up-to racing_lmpc_launch
	else
		colcon build --cmake-args -DCMAKE_BUILD_TYPE=Debug --packages-up-to ${PACKAGES}
	fi

.PHONY: build
build:
	@PACKAGES="${PACKAGES}"
	source ./tools/scripts/source_all.sh
	if [ -z "$${PACKAGES}" ] ; then
		colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-up-to racing_lmpc_launch
	else
		colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-up-to ${PACKAGES}
	fi

.PHONY: build-select
build-select:
	@PACKAGES="${PACKAGES}"
	source ./tools/scripts/source_all.sh
	colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select ${PACKAGES}

.PHONY: build-select-debug
build-select-debug:
	@PACKAGES="${PACKAGES}"
	source ./tools/scripts/source_all.sh
	colcon build --cmake-args -DCMAKE_BUILD_TYPE=Debug --packages-select ${PACKAGES}

.PHONY: reformat
reformat:
	source ./tools/scripts/source_all.sh
	autoflake --in-place --remove-unused-variables --remove-all-unused-imports --ignore-init-module-imports -r ${PATHS}
	black -l 99 ${PATHS}
	ament_uncrustify --reformat ${PATHS}

.PHONY: rosdep-install
rosdep-install:
	source ./tools/scripts/source_all.sh
	sudo apt update
	rosdep update --include-eol-distros
	rosdep install -y -r --rosdistro ${ROS_DISTRO} --ignore-src --from-paths .

.PHONY: clean-jit
clean-jit:
	find . -type f -name 'jit_*'
	@read -p "The above file(s) will be removed. Proceed? [y/N] " answer; \
	if [ "$$answer" != "${answer#[Yy]}" ] ; then \
		echo "Removing files..."; \
		find . -type f -name 'jit_*' -exec echo "Removing: {}" \; -exec rm {} \;; \
	else \
		echo "Operation canceled."; \
	fi