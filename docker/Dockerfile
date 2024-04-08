#############################################################################
#                              STANDARD IMAGE                               #
#############################################################################

FROM lucianiusw/multimodal_toolkit:base-cuda12.1.1-pytorch2.1.0-python3.10.12

# ==================================================================
# MKL https://software.intel.com/en-us/mkl
# ------------------------------------------------------------------
RUN cd /tmp && wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
    sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list' && \
    apt-get update && DEBIAN_FRONTEND=noninteractive $APT_INSTALL intel-mkl-64bit-2020.4-912

# just in case for visibility
ENV MKLROOT="/opt/intel/mkl"

COPY requirements.txt /tmp/
RUN source activate ${CONDA_ENV_NAME} && \
    # install python libraries
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    # instrall triton
    pip install --no-cache-dir  -U --pre triton && \
    # install jax from source
    pip install --no-cache-dir --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    # install flax
    pip install --no-cache-dir flax[all] && \
    # install transformers from source
    pip install --no-cache-dir git+https://github.com/huggingface/transformers

# ==================================================================
# clean up everything
# ------------------------------------------------------------------
RUN apt-get clean && \
    apt-get -y autoremove && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

# Set environment variables
ENV OMP_NUM_THREADS=20
ENV NUMEXPR_MAX_THREADS=20
ENV HYDRA_FULL_ERROR=1
ENV NCCL_DEBUG='INFO'
ENV TORCH_DISTRIBUTED_DEBUG='INFO'