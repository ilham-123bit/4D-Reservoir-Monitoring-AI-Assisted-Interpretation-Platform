"""Controlled live-test evidence only: constants, no dataset/volume/file access."""
from .grounding import FindingEvidence


def synthetic_cases():
    specs = [
        ('high-nrms',160.0,48,9,'limited','unassessed_or_consistent',4,4),
        ('coherent-weak',20.0,24,3,'limited','unassessed_or_consistent',0,0),
        ('conflicting-methods',75.0,32,9,'adequate','conflicting',12,0),
        ('missing-well-no-fluid-proof',65.0,16,9,'adequate','unassessed_or_consistent',4,2),
    ]
    packets=[]
    for name,nrms,cells,support,quality,agreement,stat_flags,ml_flags in specs:
        finding='synthetic-'+name
        facts=[]
        def fact(key,category,value):
            facts.append({'id':finding+':'+key,'name':key,'category':category,'value':value,
                          'source':'controlled synthetic fixture; not Challenge 3'})
        for key,value in [('baseline_amplitude',10.0),('monitor_amplitude',12.0),('source_indices',[12,24,30])]:fact(key,'observation',value)
        for key,value in [('mean_signed_difference',2.0),('mean_absolute_difference',2.0),
                          ('mean_rms_baseline',10.0),('mean_rms_monitor',12.0),('mean_nrms_percent',nrms),
                          ('minimum_paired_support',support),('candidate_membership',cells),
                          ('grid_extent',[[10,15],[20,27]])]:fact(key,'calculated_attribute',value)
        fact('statistical_flags','statistical_detection',stat_flags)
        fact('ml_flags','ml_detection',ml_flags)
        fact('support_quality','limitation',quality)
        fact('method_agreement','detection',agreement)
        fact('method_dependence','limitation','Shared correlated attributes; no independent confirmation')
        fact('alternative_causes','hypothesis','Acquisition/processing/scaling/noise/registration remain possible')
        fact('physical_registration','limitation','unverified')
        fact('well_association','limitation','unavailable: no CRS, datum or time-depth tie')
        packets.append(FindingEvidence(finding,'synthetic-run-'+name,tuple(facts),(
            'All values are controlled synthetic summaries, not actual Challenge 3 measurements.',
            'Physical registration and well correlation are unresolved.',
            'Windowed NRMS is independent of the isolated center amplitudes in this constructed example.',
            'No fluid movement, pressure depletion, unswept oil or recovery opportunity is established.'),
            data_classification='synthetic'))
    return packets
