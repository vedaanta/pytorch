# Owner(s): ["module: sdpa"]

import os
from unittest import mock

import torch.nn.attention as attention
from torch.nn.attention import _registry
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class FakeHandle:
    def remove(self):
        pass


class TestFlashAttentionRegistry(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        self._saved_impls = dict(_registry._FLASH_ATTENTION_IMPLS)
        self._saved_active = attention.current_flash_attention_impl()
        _registry._FLASH_ATTENTION_IMPLS.clear()
        _registry._FLASH_ATTENTION_ACTIVE = None
        self._saved_pending = _registry._FLASH_ATTENTION_PENDING_ENV
        _registry._FLASH_ATTENTION_PENDING_ENV = None

    def tearDown(self):
        _registry._FLASH_ATTENTION_IMPLS.clear()
        _registry._FLASH_ATTENTION_IMPLS.update(self._saved_impls)
        _registry._FLASH_ATTENTION_ACTIVE = self._saved_active
        _registry._FLASH_ATTENTION_PENDING_ENV = self._saved_pending
        super().tearDown()

    def test_register_and_activate_impl(self):
        calls: dict[str, bool] = {}

        def fake_register():
            calls["called"] = True
            return FakeHandle()

        attention.register_flash_attention_impl("TEST_FA", register_fn=fake_register)
        self.assertIn("TEST_FA", attention.list_flash_attention_impls())

        attention.activate_flash_attention_impl("TEST_FA")

        self.assertTrue(calls.get("called", False))
        self.assertEqual("TEST_FA", attention.current_flash_attention_impl())

    def test_activate_unknown_impl_errors(self):
        with self.assertRaisesRegex(
            ValueError, "Unknown flash attention impl 'missing'"
        ):
            attention.activate_flash_attention_impl("missing")

    def test_activate_unknown_impl_keeps_current_impl(self):
        """Asking for an unregistered impl must not deactivate the active one."""
        removes: list[str] = []

        class TrackedHandle:
            def remove(self):
                removes.append("removed")

        attention.register_flash_attention_impl(
            "GOOD", register_fn=lambda: TrackedHandle()
        )
        attention.activate_flash_attention_impl("GOOD")

        with self.assertRaisesRegex(ValueError, "Unknown flash attention impl"):
            attention.activate_flash_attention_impl("missing")

        self.assertEqual("GOOD", attention.current_flash_attention_impl())
        self.assertEqual([], removes)  # never torn down

    def test_failed_activation_reactivates_previous_impl(self):
        """A register_fn that raises (e.g. the provider package is not
        installed) must leave the previously active impl active, not silently
        drop the process to the default implementation."""
        good_registrations: list[str] = []

        class FakeGoodHandle:
            def remove(self):
                pass

        def good_register():
            good_registrations.append("registered")
            return FakeGoodHandle()

        def bad_register():
            raise ImportError("provider package is not installed")

        attention.register_flash_attention_impl("GOOD", register_fn=good_register)
        attention.register_flash_attention_impl("BAD", register_fn=bad_register)

        attention.activate_flash_attention_impl("GOOD")
        self.assertEqual("GOOD", attention.current_flash_attention_impl())

        with self.assertRaisesRegex(ImportError, "provider package"):
            attention.activate_flash_attention_impl("BAD")

        self.assertEqual("GOOD", attention.current_flash_attention_impl())
        # GOOD was re-registered as part of the rollback
        self.assertEqual(2, len(good_registrations))

    def test_activation_without_handle_is_tracked(self):
        """An impl whose register_fn returns no handle is still the current
        impl, and restoring afterwards neither warns nor raises."""
        attention.register_flash_attention_impl("NOHANDLE", register_fn=lambda: None)
        attention.activate_flash_attention_impl("NOHANDLE")
        self.assertEqual("NOHANDLE", attention.current_flash_attention_impl())

        attention.restore_flash_attention_impl()
        self.assertIsNone(attention.current_flash_attention_impl())

    def test_env_var_activates_registered_impl(self):
        """TORCH_ATTENTION_IMPL names an impl that is already registered."""
        attention.register_flash_attention_impl(
            "ENVIMPL", register_fn=lambda: FakeHandle()
        )
        with mock.patch.dict(os.environ, {_registry.ENV_VAR: "ENVIMPL"}):
            _registry._activate_from_env()
        self.assertEqual("ENVIMPL", attention.current_flash_attention_impl())

    def test_env_var_defers_until_impl_registers(self):
        """An out-of-tree impl is not registered when torch.nn.attention is
        imported, so the request waits for its registration."""
        with mock.patch.dict(os.environ, {_registry.ENV_VAR: "LATEIMPL"}):
            _registry._activate_from_env()
        self.assertIsNone(attention.current_flash_attention_impl())
        self.assertEqual("LATEIMPL", _registry._FLASH_ATTENTION_PENDING_ENV)

        attention.register_flash_attention_impl(
            "LATEIMPL", register_fn=lambda: FakeHandle()
        )
        self.assertEqual("LATEIMPL", attention.current_flash_attention_impl())
        self.assertIsNone(_registry._FLASH_ATTENTION_PENDING_ENV)

    def test_env_var_registration_of_other_impl_does_not_activate(self):
        """Only the requested name activates; a different provider importing
        first must not be hijacked by the pending request."""
        with mock.patch.dict(os.environ, {_registry.ENV_VAR: "WANTED"}):
            _registry._activate_from_env()
        attention.register_flash_attention_impl(
            "OTHER", register_fn=lambda: FakeHandle()
        )
        self.assertIsNone(attention.current_flash_attention_impl())
        self.assertEqual("WANTED", _registry._FLASH_ATTENTION_PENDING_ENV)

    def test_env_var_failed_activation_does_not_raise(self):
        """A provider whose register_fn raises must not break the import that
        triggered the deferred activation."""

        def _boom():
            raise RuntimeError("provider is broken")

        with mock.patch.dict(os.environ, {_registry.ENV_VAR: "BROKEN"}):
            _registry._activate_from_env()
        attention.register_flash_attention_impl("BROKEN", register_fn=_boom)
        self.assertIsNone(attention.current_flash_attention_impl())

    def test_env_var_unset_or_blank_is_a_no_op(self):
        for value in ("", "   "):
            with mock.patch.dict(os.environ, {_registry.ENV_VAR: value}):
                _registry._activate_from_env()
            self.assertIsNone(attention.current_flash_attention_impl())
            self.assertIsNone(_registry._FLASH_ATTENTION_PENDING_ENV)


if __name__ == "__main__":
    run_tests()
