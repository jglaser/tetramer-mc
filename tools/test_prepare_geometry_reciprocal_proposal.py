"""Ensure geometry provenance cannot quietly become a fitted native proposal."""
import copy
import unittest
from prepare_geometry_reciprocal_proposal import (
    BASE, GENERATION, SHAPE, read, validate_construction, reciprocal_envelope,
)


class GeometryPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base,cls.generation,cls.shape=read(BASE),read(GENERATION),read(SHAPE)

    def test_archived_geometry_covariances_reconstruct_without_native_data(self):
        value=validate_construction(self.base,self.generation,self.shape)
        self.assertTrue(value['passed']);self.assertLess(value['maximum_covariance_reconstruction_error'],1e-10)
        self.assertEqual(value['native_geometry_inputs'],0)

    def test_native_angular_scale_or_reweighted_components_fail(self):
        for choice in ('angular_length','weights','means'):
            base=copy.deepcopy(self.base)
            if choice=='angular_length':base[choice]*=1.1
            elif choice=='weights':base[choice][0]*=2
            else:base[choice][0][0]=.1
            with self.subTest(choice=choice),self.assertRaises(ValueError):
                validate_construction(base,self.generation,self.shape)

    def test_covariance_fit_or_contact_anchor_change_fails(self):
        for choice in ('covariance','anchor'):
            base=copy.deepcopy(self.base)
            if choice=='covariance':base['covariances'][0][0][0]+=.1
            else:base['anchors'][0]['position'][0]+=.1
            with self.subTest(choice=choice),self.assertRaises(ValueError):
                validate_construction(base,self.generation,self.shape)

    def test_reciprocal_envelope_keeps_entire_historical_model(self):
        wrapped=reciprocal_envelope(self.base)
        self.assertEqual(wrapped['base_model'],self.base)
        self.assertIsNot(wrapped['base_model'],self.base)
        self.assertEqual(wrapped['reciprocal_components'],[True]*96)


if __name__=='__main__':unittest.main()
