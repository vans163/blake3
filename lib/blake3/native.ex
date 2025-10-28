defmodule Blake3.Native do
  @moduledoc """
  Blake3.Native is the rustler module that will be replaced with the nif rust functions.
  This module doesn't need to be called direcly.
  """

  use Rustler,
    otp_app: :blake3_ex

  def hash(_str), do: error()
  def new(), do: error()
  def update(_state, _str), do: error()
  def update_rayon(_state, _str), do: error()
  def finalize(_state), do: error()
  def finalize_xof(_state, _size), do: error()
  def derive_key(_context, _key), do: error()
  def keyed_hash(_key, _str), do: error()
  def new_keyed(_key), do: error()
  def reset(_state), do: error()
  def freivalds(_tensor), do: error()
  def freivalds_e260(_tensor, _vr), do: error()

  defp error, do: :erlang.nif_error(:nif_not_loaded)
end
